import json
import os
import re
import time
import random
import asyncio
import aiohttp
from tqdm import tqdm
from collections import Counter
import logging
import traceback
from asyncio import Lock

# ---------- CONFIG ----------
MODE = "criminal"  # "civil" or "criminal"

CIVIL_INPUT    = r"C:\Users\Kushal\Desktop\new\Data_Gen\gen_civil.json"
CIVIL_OUTPUT   = r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_civil.json"
CIVIL_TRACKING = r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_tracking_civil.json"
CIVIL_LOG      = r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_log_civil.txt"

CRIMINAL_INPUT    = r"C:\Users\Kushal\Desktop\new\Data_Gen\gen_criminal.json"
CRIMINAL_OUTPUT   = r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_criminal.json"
CRIMINAL_TRACKING = r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_tracking_criminal.json"
CRIMINAL_LOG      = r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_log_criminal.txt"

API_KEYS = [
    'AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM',
    'AIzaSyCz41qZibYl1rca9Ye7_pkKaaBlQnms6ME',
    'AIzaSyA27TjinZixsBBl1ZkY7dRV9dK5Hf-ZzbA',
    'AIzaSyBFEIHo4GKXri8a3rNAaQKq0EwRSrwjwYs',
    'AIzaSyCAEEIT9lc5WxLAyq_N4SaKouAggntDxC4'
]
GEMINI_MODEL = "gemini-2.5-flash-lite"

MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 3
BATCH_MAX_ITEMS = 5
BATCH_MAX_CHARS = 14000
RETRY_CYCLES = 5
BACKOFF_BASE = 2
RATE_LIMIT_DELAY = 0.3  # seconds between batch requests
TARGET_FAST = 8  # seconds per batch → speed up if faster
TARGET_SLOW = 15  # seconds per batch → slow down if slower
MAX_INDIVIDUAL_RETRIES = 3  # Max retries for individual cases

# ---------- LOGGING SETUP ----------
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger

# ---------- PROMPT ----------
POLISH_PROMPT = """
You are an expert legal data verifier.
You will be given an array of JSON legal case objects.
Your task is to polish them while keeping the structure intact.

{
  "case_id": "string",
  "instruction": "string",
  "question": "string",
  "analysis": {
    "case_summary": {
      "facts": "string",
      "legal_issue": "string",
      "law_applied": "string",
      "judgment": "string"
    },
    "court_reasoning": {
      "reasoning": "string",
      "decision": "string"
    }
  },
  "metadata": {
    "category": "string",
    "case_type": "string",
    "law": "string"
  }
}

Rules:
- Return only a valid JSON array. No explanations outside JSON.
- Do not add or remove keys. Only update values if factually wrong, inconsistent, or unclear.
- Fix contradictions between facts, reasoning, law_applied, and judgment.
- Do not invent new laws or cases. Use only provided input.
- If information is missing, leave the field as-is.
- Ensure "reasoning" is concise, logical, and consistent.
- Replace unnecessary newlines with spaces.
- Keep text clear and formal.
- If the case cannot be parsed or validated, return the object with the same case_id and add "status": "invalid".
"""

# ---------- HELPERS ----------
def clean_json_response(raw_text):
    """Extract and clean JSON from API response"""
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"```$", "", raw_text)
    
    # Try to find JSON array first
    m = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if m:
        return m.group(0)
    
    # Fallback to JSON object
    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if m:
        return m.group(0)
    
    return raw_text

def normalize_case(input_case, output_case):
    """Normalize case structure ensuring all required fields"""
    return {
        "case_id": input_case.get("case_id", ""),
        "instruction": output_case.get("instruction") or input_case.get("instruction", "Summarize the case and also answer the legal question."),
        "question": output_case.get("question") or input_case.get("question", f"What did the court decide in relation to {input_case.get('legal_issue','')}?"),
        "analysis": {
            "case_summary": {
                "facts": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("facts")
                    or input_case.get("analysis", {}).get("case_summary", {}).get("facts", "")
                ),
                "legal_issue": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("legal_issue")
                    or input_case.get("analysis", {}).get("case_summary", {}).get("legal_issue", "")
                ),
                "law_applied": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("law_applied")
                    or input_case.get("analysis", {}).get("case_summary", {}).get("law_applied", "")
                ),
                "judgment": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("judgment")
                    or input_case.get("analysis", {}).get("case_summary", {}).get("judgment", "")
                )
            },
            "court_reasoning": {
                "reasoning": (
                    output_case.get("analysis", {})
                               .get("court_reasoning", {})
                               .get("reasoning")
                    or input_case.get("analysis", {}).get("court_reasoning", {}).get("reasoning", "")
                ),
                "decision": (
                    output_case.get("analysis", {})
                               .get("court_reasoning", {})
                               .get("decision")
                    or input_case.get("analysis", {}).get("court_reasoning", {}).get("decision", "")
                )
            }
        },
        "metadata": {
            "category": input_case.get("metadata", {}).get("category", ""),
            "case_type": input_case.get("metadata", {}).get("case_type", ""),
            "law": input_case.get("metadata", {}).get("law", "")
        }
    }

def load_json_file(path, default, logger):
    """Safely load JSON file with error handling"""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                count = len(data) if isinstance(data, (list, dict)) else 0
                logger.info(f"✅ Loaded {count} items from {path}")
                return data
        else:
            logger.info(f"📁 File {path} doesn't exist, using default")
            return default
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error in {path}: {e}")
        return default
    except Exception as e:
        logger.error(f"❌ Error loading {path}: {e}")
        return default

def save_json_file(path, data, logger):
    """Safely save JSON file with error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Atomic rename
        os.replace(temp_path, path)
        
        count = len(data) if isinstance(data, (list, dict)) else 0
        logger.debug(f"💾 Saved {count} items to {path}")
        
    except Exception as e:
        logger.error(f"❌ Error saving {path}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(path + ".tmp"):
            os.remove(path + ".tmp")

def print_tracking_summary(tracking, logger):
    """Print detailed tracking statistics"""
    if not tracking:
        logger.info("📊 No tracking data available")
        return
        
    success_count = sum(1 for v in tracking.values() if v.get("status") == "success")
    fail_count = sum(1 for v in tracking.values() if v.get("status") in ["fail", "failed"])
    invalid_count = sum(1 for v in tracking.values() if v.get("status") == "invalid")
    total_tracked = len(tracking)
    
    logger.info("📊 Tracking Summary:")
    logger.info(f"   Total Tracked Cases: {total_tracked}")
    logger.info(f"   ✅ Successful: {success_count}")
    logger.info(f"   ❌ Failed: {fail_count}")
    logger.info(f"   ⚠️ Invalid: {invalid_count}")
    
    if fail_count > 0:
        logger.info("   Failed Case Details:")
        fail_cases = [(cid, info) for cid, info in tracking.items() if info.get("status") in ["fail", "failed"]]
        for cid, info in fail_cases[:5]:  # Show only first 5 failed cases
            logger.info(f"      Case {cid}: {info.get('reason', 'No reason provided')}")
        if len(fail_cases) > 5:
            logger.info(f"      ... and {len(fail_cases) - 5} more failed cases")

# ---------- THREAD-SAFE FILE MANAGER ----------
class FileManager:
    """Thread-safe file operations manager"""
    def __init__(self, logger):
        self.logger = logger
        self.lock = Lock()
    
    async def update_results_and_tracking(self, results_file, tracking_file, new_results, new_tracking):
        """Atomically update both results and tracking files"""
        async with self.lock:
            try:
                # Load current data
                current_results = load_json_file(results_file, [], self.logger)
                current_tracking = load_json_file(tracking_file, {}, self.logger)
                
                # Update with new data
                if isinstance(current_results, list) and isinstance(new_results, list):
                    current_results.extend(new_results)
                
                if isinstance(current_tracking, dict) and isinstance(new_tracking, dict):
                    current_tracking.update(new_tracking)
                
                # Save atomically
                save_json_file(results_file, current_results, self.logger)
                save_json_file(tracking_file, current_tracking, self.logger)
                
                return current_results, current_tracking
                
            except Exception as e:
                self.logger.error(f"❌ Error updating files: {e}")
                return None, None

# ---------- IMPROVED API KEY MANAGER ----------
class APIKeyManager:
    """Enhanced API key manager with better failure tracking"""
    def __init__(self, api_keys, logger):
        self.keys = [
            {
                "key": k, 
                "failures": 0, 
                "last_used": 0, 
                "requests": 0, 
                "successes": 0,
                "consecutive_failures": 0,
                "cooldown_until": 0
            } 
            for k in api_keys
        ]
        self.lock = asyncio.Lock()
        self.logger = logger
        self.logger.info(f"🔑 Initialized {len(self.keys)} API keys")

    async def get_key(self):
        """Get the best available API key"""
        async with self.lock:
            current_time = time.time()
            
            # Filter out keys in cooldown
            available = [
                k for k in self.keys 
                if k["failures"] < RETRY_CYCLES 
                and k["cooldown_until"] <= current_time
                and k["consecutive_failures"] < 3
            ]
            
            if not available:
                # Fallback to any key not in cooldown
                available = [k for k in self.keys if k["cooldown_until"] <= current_time]
                
            if not available:
                # Last resort - use the key with least recent failures
                available = [min(self.keys, key=lambda k: k["consecutive_failures"])]
                self.logger.warning("⚠️ All keys exhausted, using least failed key")
            
            # Select key with least recent usage and best success rate
            key_info = min(available, key=lambda k: (k["last_used"], -k.get("successes", 0)))
            key_info["last_used"] = current_time
            key_info["requests"] += 1
            
            self.logger.debug(f"🔑 Selected key: {key_info['key'][-8:]} (success rate: {self._get_success_rate(key_info):.1f}%)")
            return key_info["key"]

    async def record_failure(self, key):
        """Record API key failure with enhanced tracking"""
        async with self.lock:
            for k in self.keys:
                if k["key"] == key:
                    k["failures"] += 1
                    k["consecutive_failures"] += 1
                    
                    # Apply cooldown for repeated failures
                    if k["consecutive_failures"] >= 2:
                        cooldown_time = min(60, k["consecutive_failures"] * 10)  # Max 60s cooldown
                        k["cooldown_until"] = time.time() + cooldown_time
                        self.logger.warning(f"🚫 Key {key[-8:]} in cooldown for {cooldown_time}s")
                    
                    self.logger.debug(f"❌ Key {key[-8:]} failure #{k['failures']} (consecutive: {k['consecutive_failures']})")
                    break

    async def record_success(self, key):
        """Record API key success"""
        async with self.lock:
            for k in self.keys:
                if k["key"] == key:
                    k["successes"] += 1
                    k["consecutive_failures"] = 0  # Reset consecutive failures
                    k["cooldown_until"] = 0  # Remove cooldown
                    self.logger.debug(f"✅ Key {key[-8:]} success #{k['successes']}")
                    break

    def _get_success_rate(self, key_info):
        """Calculate success rate for a key"""
        return (key_info["successes"] / key_info["requests"] * 100) if key_info["requests"] > 0 else 0

    def print_stats(self):
        """Print detailed API key statistics"""
        self.logger.info("\n🔑 API Key Performance:")
        for i, k in enumerate(self.keys):
            success_rate = self._get_success_rate(k)
            cooldown_status = "🚫 COOLDOWN" if k["cooldown_until"] > time.time() else "✅ Active"
            self.logger.info(f"   Key {i+1} ({k['key'][-8:]}): {k['successes']}/{k['requests']} ({success_rate:.1f}%) - {cooldown_status}")

# ---------- ENHANCED GEMINI CALL ----------
async def call_gemini(session, entries, api_manager, logger, max_retries=RETRY_CYCLES):
    """Enhanced Gemini API call with better error handling"""
    MODEL_TRY = [GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.5-pro"]
    
    def _truncate_text(text, max_len=4000):
        """Safely truncate text fields"""
        if not isinstance(text, str) or len(text) <= max_len:
            return text
        return text[:max_len] + " ...[truncated]"
    
    # Create slimmed version for API call
    slim_entries = []
    for case in entries:
        slim_case = dict(case)
        analysis = slim_case.get("analysis", {})
        
        # Truncate case summary fields
        case_summary = analysis.get("case_summary", {})
        for field in ("facts", "law_applied", "judgment"):
            if field in case_summary:
                case_summary[field] = _truncate_text(case_summary[field])
        
        # Truncate court reasoning fields  
        court_reasoning = analysis.get("court_reasoning", {})
        for field in ("reasoning", "decision"):
            if field in court_reasoning:
                court_reasoning[field] = _truncate_text(court_reasoning[field])
        
        slim_entries.append(slim_case)

    prompt_text = f"{POLISH_PROMPT}\n\nHere is the JSON:\n{json.dumps(slim_entries, ensure_ascii=False)}"
    logger.debug(f"📤 Payload size: {len(prompt_text)} characters for {len(entries)} cases")
    
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    headers = {"Content-Type": "application/json"}

    # Try each model
    for model_name in MODEL_TRY:
        logger.debug(f"🤖 Trying model: {model_name}")
        
        # Retry with backoff for each model
        for attempt in range(max_retries):
            api_key = await api_manager.get_key()
            
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                
                timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                    body_text = await resp.text()
                    
                    if resp.status != 200:
                        logger.warning(f"⚠️ HTTP {resp.status}: {body_text[:500]}")
                        
                        if resp.status == 429:  # Rate limit
                            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                            logger.info(f"⏳ Rate limited, backing off {wait:.1f}s...")
                            await asyncio.sleep(wait)
                            await api_manager.record_failure(api_key)
                            continue
                            
                        elif 500 <= resp.status < 600:  # Server error
                            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                            logger.info(f"⏳ Server error, backing off {wait:.1f}s...")
                            await asyncio.sleep(wait)
                            await api_manager.record_failure(api_key)
                            continue
                            
                        else:  # Other HTTP errors
                            await api_manager.record_failure(api_key)
                            break  # Don't retry for client errors
                    
                    # Parse response
                    try:
                        data = json.loads(body_text)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON response: {e}")
                        await api_manager.record_failure(api_key)
                        continue
                    
                    candidates = data.get("candidates", [])
                    if not candidates:
                        logger.warning("⚠️ No candidates in response")
                        await api_manager.record_failure(api_key)
                        continue
                    
                    # Extract text from response
                    content = candidates[0].get("content", {})
                    if isinstance(content, list) and len(content) > 0:
                        content = content[0]
                    
                    parts = content.get("parts", [])
                    raw_text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    
                    if not raw_text:
                        logger.warning(f"⚠️ Empty response from {model_name}")
                        await api_manager.record_failure(api_key)
                        continue
                    
                    # Clean and parse JSON
                    cleaned_json = clean_json_response(raw_text)
                    logger.debug(f"🧹 Cleaned response preview: {cleaned_json[:200].replace(chr(10), ' ')}")
                    
                    try:
                        parsed_result = json.loads(cleaned_json)
                        
                        # Ensure result is a list
                        if isinstance(parsed_result, dict):
                            parsed_result = [parsed_result]
                        
                        if not isinstance(parsed_result, list):
                            logger.error("❌ Response is not a list or dict")
                            await api_manager.record_failure(api_key)
                            continue
                        
                        await api_manager.record_success(api_key)
                        logger.debug(f"✅ Successfully parsed {len(parsed_result)} cases")
                        return parsed_result
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON parse error: {e}")
                        logger.debug(f"📝 Raw text: {raw_text[:500]}")
                        
                        # Try to salvage partial JSON
                        salvage_match = re.search(r"\[.*\]", cleaned_json, re.DOTALL)
                        if not salvage_match:
                            salvage_match = re.search(r"\{.*\}", cleaned_json, re.DOTALL)
                        
                        if salvage_match:
                            try:
                                salvaged = json.loads(salvage_match.group(0))
                                if isinstance(salvaged, dict):
                                    salvaged = [salvaged]
                                await api_manager.record_success(api_key)
                                logger.info(f"🔧 Salvaged {len(salvaged)} cases from partial JSON")
                                return salvaged
                            except:
                                pass
                        
                        # Retry with backoff
                        wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"⏳ JSON error, retrying in {wait:.1f}s...")
                        await asyncio.sleep(wait)
                        await api_manager.record_failure(api_key)
                        continue
            
            except asyncio.TimeoutError:
                logger.error(f"⏰ Request timeout for model {model_name}")
                await api_manager.record_failure(api_key)
                continue
                
            except Exception as e:
                await api_manager.record_failure(api_key)
                logger.error(f"❌ Unexpected error: {e}")
                logger.debug(traceback.format_exc())
                
                wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"⏳ Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
                continue
        
        logger.warning(f"⚠️ Model {model_name} failed after {max_retries} attempts")
    
    logger.error("❌ All models failed")
    return None

# ---------- ENHANCED BATCH PROCESSING ----------
async def process_batch(session, batch, file_manager, output_file, tracking_file, api_manager, logger):
    """Process a batch of cases with enhanced error handling"""
    batch_ids = [case.get("case_id", "unknown") for case in batch]
    logger.info(f"🔥 Processing batch: {batch_ids}")

    start_time = time.perf_counter()
    
    # Try batch processing first
    response = await call_gemini(session, batch, api_manager, logger)
    
    batch_results = []
    batch_tracking = {}
    
    if response and isinstance(response, list) and len(response) == len(batch):
        # Batch processing successful
        logger.info(f"✅ Batch processing successful for {len(batch)} cases")
        
        for input_case, output_case in zip(batch, response):
            case_id = input_case.get("case_id")
            
            if output_case.get("status") == "invalid":
                batch_results.append({"case_id": case_id, "status": "invalid"})
                batch_tracking[case_id] = {"status": "invalid", "processed_at": time.time()}
                logger.warning(f"⚠️ Case {case_id} marked as invalid")
            else:
                try:
                    normalized_case = normalize_case(input_case, output_case)
                    batch_results.append(normalized_case)
                    batch_tracking[case_id] = {"status": "success", "processed_at": time.time()}
                    logger.debug(f"✅ Case {case_id} processed successfully")
                except Exception as e:
                    logger.error(f"❌ Error normalizing case {case_id}: {e}")
                    batch_results.append(input_case)  # Keep original
                    batch_tracking[case_id] = {"status": "fail", "reason": f"Normalization error: {e}", "processed_at": time.time()}
    
    else:
        # Batch failed - process individually
        logger.warning(f"❌ Batch failed, processing {len(batch)} cases individually")
        
        for case in batch:
            case_id = case.get("case_id")
            individual_success = False
            
            # Try processing individual case with retries
            for retry in range(MAX_INDIVIDUAL_RETRIES):
                try:
                    individual_response = await call_gemini(session, [case], api_manager, logger, max_retries=2)
                    
                    if individual_response and len(individual_response) > 0:
                        output_case = individual_response[0]
                        
                        if output_case.get("status") == "invalid":
                            batch_results.append({"case_id": case_id, "status": "invalid"})
                            batch_tracking[case_id] = {"status": "invalid", "processed_at": time.time()}
                            logger.warning(f"⚠️ Individual case {case_id} marked as invalid")
                        else:
                            normalized_case = normalize_case(case, output_case)
                            batch_results.append(normalized_case)
                            batch_tracking[case_id] = {"status": "success", "processed_at": time.time()}
                            logger.info(f"✅ Individual case {case_id} processed successfully")
                        
                        individual_success = True
                        break
                        
                except Exception as e:
                    logger.error(f"❌ Individual processing error for {case_id} (retry {retry+1}): {e}")
                    if retry < MAX_INDIVIDUAL_RETRIES - 1:
                        await asyncio.sleep(RATE_LIMIT_DELAY * (retry + 1))
            
            if not individual_success:
                batch_results.append(case)  # Keep original case
                batch_tracking[case_id] = {
                    "status": "fail", 
                    "reason": "Individual processing failed after retries",
                    "processed_at": time.time()
                }
                logger.error(f"❌ Case {case_id} failed after {MAX_INDIVIDUAL_RETRIES} individual attempts")

    # Save results atomically
    await file_manager.update_results_and_tracking(output_file, tracking_file, batch_results, batch_tracking)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    success_count = sum(1 for tracking in batch_tracking.values() if tracking.get("status") == "success")
    logger.info(f"⚡ Batch {batch_ids} completed in {elapsed:.2f}s ({success_count}/{len(batch)} successful)")
    
    return elapsed

# ---------- MAIN PIPELINE ----------
async def process_cases(input_file, output_file, tracking_file, log_file):
    """Enhanced main processing pipeline"""
    logger = setup_logging(log_file)
    logger.info("🚀 Starting Enhanced Legal Case Polishing Pipeline")

    # Log processing configuration
    logger.info("📋 Processing Configuration:")
    logger.info(f"   - Input: {input_file}")
    logger.info(f"   - Output: {output_file}")
    logger.info(f"   - Tracking: {tracking_file}")
    logger.info(f"   - Log: {log_file}")
    logger.info(f"   - Batch Size: {BATCH_MAX_ITEMS} cases")
    logger.info(f"   - Max Payload: {BATCH_MAX_CHARS} chars")
    logger.info(f"   - Concurrency: {MIN_CONCURRENCY}-{MAX_CONCURRENCY} (adaptive)")
    logger.info(f"   - API Keys: {len(API_KEYS)}")
    logger.info(f"   - Rate Limit Delay: {RATE_LIMIT_DELAY}s")

    # Load input data
    all_cases = load_json_file(input_file, [], logger)
    if not all_cases:
        logger.error("❌ No cases found in input file")
        return

    # Load tracking data
    tracking = load_json_file(tracking_file, {}, logger)
    
    # Initialize file manager for thread-safe operations
    file_manager = FileManager(logger)

    # Categorize cases based on tracking status
    failed_cases = []
    new_cases = []
    completed_cases = 0

    for case in all_cases:
        case_id = case.get("case_id", "")
        if not case_id:
            logger.warning("⚠️ Found case without case_id, skipping")
            continue
            
        status = tracking.get(case_id, {}).get("status", "")
        
        if status == "success" or status == "invalid":
            completed_cases += 1
        elif status == "fail" or status == "failed":  # Handle both formats
            failed_cases.append(case)
        else:
            new_cases.append(case)

    # Prioritize failed cases, then new cases
    cases_to_process = failed_cases + new_cases
    total_cases = len(all_cases)
    to_process_count = len(cases_to_process)

    # Log case breakdown
    logger.info("📊 Case Status Breakdown:")
    logger.info(f"   Total cases: {total_cases}")
    logger.info(f"   ✅ Completed (success + invalid): {completed_cases}")
    logger.info(f"   ❌ Failed (to retry): {len(failed_cases)}")
    logger.info(f"   🆕 New cases: {len(new_cases)}")
    logger.info(f"   📋 To process: {to_process_count}")

    if to_process_count == 0:
        logger.info("✅ All cases already processed successfully!")
        print_tracking_summary(tracking, logger)
        return

    # Initialize processing components
    api_manager = APIKeyManager(API_KEYS, logger)
    total_start_time = time.perf_counter()
    
    # Adaptive concurrency control
    current_concurrency = MIN_CONCURRENCY
    processed_cases = 0
    batch_index = 0

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2),
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        
        with tqdm(total=to_process_count, desc="Processing cases", unit="case") as pbar:
            
            while processed_cases < to_process_count:
                batch_start_idx = processed_cases
                concurrent_batches = []
                batch_case_counts = []
                
                # Create concurrent batches
                for _ in range(current_concurrency):
                    if batch_start_idx >= to_process_count:
                        break
                    
                    # Create batch respecting size limits
                    batch_end_idx = min(batch_start_idx + BATCH_MAX_ITEMS, to_process_count)
                    batch = cases_to_process[batch_start_idx:batch_end_idx]
                    
                    # Check if batch is too large (character-wise)
                    batch_json = json.dumps(batch, ensure_ascii=False)
                    if len(batch_json) > BATCH_MAX_CHARS and len(batch) > 1:
                        # Split oversized batch
                        logger.info(f"📦 Splitting oversized batch ({len(batch_json)} chars) into individual cases")
                        for individual_case in batch:
                            concurrent_batches.append([individual_case])
                            batch_case_counts.append(1)
                    else:
                        concurrent_batches.append(batch)
                        batch_case_counts.append(len(batch))
                    
                    batch_start_idx = batch_end_idx

                if not concurrent_batches:
                    break

                total_batch_cases = sum(batch_case_counts)
                logger.info(f"🔥 Starting round {batch_index + 1}: {len(concurrent_batches)} batches, {total_batch_cases} cases")

                # Process batches concurrently with rate limiting
                round_start_time = time.perf_counter()
                batch_tasks = []
                
                for i, batch in enumerate(concurrent_batches):
                    if i > 0:  # Add delay between batch starts
                        await asyncio.sleep(RATE_LIMIT_DELAY)
                    
                    task = asyncio.create_task(
                        process_batch(session, batch, file_manager, output_file, tracking_file, api_manager, logger)
                    )
                    batch_tasks.append(task)

                # Wait for all batches to complete
                batch_durations = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                round_end_time = time.perf_counter()
                round_duration = round_end_time - round_start_time

                # Update progress
                processed_cases += total_batch_cases
                pbar.update(total_batch_cases)
                batch_index += 1

                # Load updated tracking for statistics
                current_tracking = load_json_file(tracking_file, {}, logger)
                success_count = sum(1 for v in current_tracking.values() if v.get("status") == "success")
                fail_count = sum(1 for v in current_tracking.values() if v.get("status") == "fail")
                invalid_count = sum(1 for v in current_tracking.values() if v.get("status") == "invalid")

                logger.info(f"⚡ Round {batch_index} completed in {round_duration:.2f}s")
                logger.info(f"📊 Progress: {processed_cases}/{to_process_count} processed, {success_count} successful, {fail_count} failed, {invalid_count} invalid")

                # Adaptive concurrency adjustment
                avg_batch_duration = sum(d for d in batch_durations if isinstance(d, (int, float))) / len([d for d in batch_durations if isinstance(d, (int, float))])
                
                if avg_batch_duration < TARGET_FAST and current_concurrency < MAX_CONCURRENCY:
                    current_concurrency += 1
                    logger.info(f"⬆️ Increasing concurrency → {current_concurrency}")
                elif avg_batch_duration > TARGET_SLOW and current_concurrency > MIN_CONCURRENCY:
                    current_concurrency -= 1
                    logger.info(f"⬇️ Decreasing concurrency → {current_concurrency}")

                # Small delay between rounds
                if processed_cases < to_process_count:
                    await asyncio.sleep(RATE_LIMIT_DELAY)

    # Final statistics
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time

    # Load final tracking data
    final_tracking = load_json_file(tracking_file, {}, logger)
    final_success = sum(1 for v in final_tracking.values() if v.get("status") == "success")
    final_fail = sum(1 for v in final_tracking.values() if v.get("status") == "fail")
    final_invalid = sum(1 for v in final_tracking.values() if v.get("status") == "invalid")
    final_completed = final_success + final_invalid

    success_rate = (final_success / total_cases * 100) if total_cases > 0 else 0
    completion_rate = (final_completed / total_cases * 100) if total_cases > 0 else 0

    logger.info("\n🎯 Final Pipeline Results:")
    logger.info(f"   Total input cases: {total_cases}")
    logger.info(f"   Cases processed this run: {to_process_count}")
    logger.info(f"   ✅ Total successful: {final_success} ({success_rate:.1f}%)")
    logger.info(f"   ⚠️ Total invalid: {final_invalid}")
    logger.info(f"   ❌ Total failed: {final_fail}")
    logger.info(f"   🎯 Total completed: {final_completed} ({completion_rate:.1f}%)")
    logger.info(f"   ⏱️ Total pipeline time: {total_duration:.2f} seconds")
    
    if to_process_count > 0:
        avg_time_per_case = total_duration / to_process_count
        logger.info(f"   ⚡ Average time per case: {avg_time_per_case:.2f} seconds")

    # Print detailed tracking summary
    print_tracking_summary(final_tracking, logger)
    
    # Print API key performance
    api_manager.print_stats()

    # Success/completion thresholds
    if completion_rate >= 95:
        logger.info("🌟 Excellent! Over 95% of cases completed successfully")
    elif completion_rate >= 90:
        logger.info("👍 Good! Over 90% of cases completed")
    elif completion_rate >= 80:
        logger.info("⚠️ Acceptable, but {:.1f}% of cases still need attention".format(100 - completion_rate))
    else:
        logger.warning("🚨 Many cases failed - please check API keys and input data quality")

# ---------- GRACEFUL SHUTDOWN HANDLER ----------
async def graceful_shutdown(input_file, output_file, tracking_file, log_file, logger):
    """Handle graceful shutdown and save current state"""
    logger.info("🛑 Graceful shutdown initiated...")
    
    try:
        # Load and save current state
        results = load_json_file(output_file, [], logger)
        tracking = load_json_file(tracking_file, {}, logger)
        
        save_json_file(output_file, results, logger)
        save_json_file(tracking_file, tracking, logger)
        
        logger.info("💾 Current state saved successfully")
        print_tracking_summary(tracking, logger)
        
    except Exception as e:
        logger.error(f"❌ Error during graceful shutdown: {e}")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    # Select files based on MODE
    if MODE == "civil":
        input_file = CIVIL_INPUT
        output_file = CIVIL_OUTPUT
        tracking_file = CIVIL_TRACKING
        log_file = CIVIL_LOG
    elif MODE == "criminal":
        input_file = CRIMINAL_INPUT
        output_file = CRIMINAL_OUTPUT
        tracking_file = CRIMINAL_TRACKING
        log_file = CRIMINAL_LOG
    else:
        print("❌ Invalid MODE. Must be 'civil' or 'criminal'")
        exit(1)

    try:
        asyncio.run(process_cases(input_file, output_file, tracking_file, log_file))
        
    except KeyboardInterrupt:
        logger = setup_logging(log_file)
        logger.info("⛔ Processing interrupted by user")
        asyncio.run(graceful_shutdown(input_file, output_file, tracking_file, log_file, logger))
        
    except Exception as e:
        logger = setup_logging(log_file)
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        asyncio.run(graceful_shutdown(input_file, output_file, tracking_file, log_file, logger))