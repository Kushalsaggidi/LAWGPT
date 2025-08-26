import json
import os
import re
import time
import random
import asyncio
import aiohttp
from tqdm import tqdm
from pathlib import Path
import logging
import traceback
from asyncio import Lock

# ---------- CONFIG ----------
MODE = "criminal"  # "civil" or "criminal"

INPUT_FILES = {
    "civil": Path(r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_civil.json"),
    "criminal": Path(r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_criminal.json")
}

OUTPUT_FILES = {
    "civil": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\verified_civil.json"),
    "criminal": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\verified_criminal.json")
}

TRACKING_FILES = {
    "civil": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\tracking_civil.json"),
    "criminal": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\tracking_criminal.json")
}

CHECKPOINT_FILES = {
    "civil": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\checkpoint_civil.json"),
    "criminal": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\checkpoint_criminal.json")
}

LOG_FILES = {
    "civil": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\verify_log_civil.txt"),
    "criminal": Path(r"C:\Users\Kushal\Desktop\new\Data_Verify\verify_log_criminal.txt")
}

API_KEYS = [
    'AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM',
    'AIzaSyCz41qZibYl1rca9Ye7_pkKaaBlQnms6ME',
    'AIzaSyA27TjinZixsBBl1ZkY7dRV9dK5Hf-ZzbA',
    'AIzaSyBFEIHo4GKXri8a3rNAaQKq0EwRSrwjwYs',
    'AIzaSyCAEEIT9lc5WxLAyq_N4SaKouAggntDxC4'
]

GEMINI_MODEL = "gemini-2.5-flash-lite"
MODEL_VARIANTS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]

MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 3
BATCH_MAX_ITEMS = 5
BATCH_MAX_CHARS = 12000
RETRY_CYCLES = 5
BACKOFF_BASE = 2
RATE_LIMIT_DELAY = 0.3
TARGET_FAST = 8
TARGET_SLOW = 15
MAX_INDIVIDUAL_RETRIES = 3

# ---------- LOGGING SETUP ----------
def setup_logging(log_file):
    os.makedirs(log_file.parent, exist_ok=True)
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

# ---------- VERIFICATION PROMPT ----------
VERIFY_PROMPT = """
You are a legal case validator. Evaluate each legal case for validity and consistency.

For each case, respond with exactly one line:
KEEP - if the case has sound legal reasoning and consistency  
REMOVE: [brief reason] - if the case has major flaws

VALIDATION CRITERIA:
1. Question-Analysis Alignment: Does the analysis actually answer the legal question asked? (Including negative outcomes, withdrawals, dismissals, etc.)
2. Facts-Legal Issue Consistency: Do the facts logically lead to the stated legal issue?
3. Law Application Accuracy: Is the cited law correctly applied to the legal issue?
4. Court Reasoning Logic: Does the reasoning follow legal principles and connect to the facts?
5. Judgment Consistency: Does the final judgment align with the reasoning provided?
6. Metadata Accuracy: Do category, case_type, and law match the case content?

KEEP cases that:
- Have coherent legal logic flow (facts → issue → law → reasoning → judgment)
- Correctly cite and apply relevant laws
- Provide sound court reasoning that addresses the legal question
- Show consistency between all components
- **Include valid procedural outcomes (withdrawals, dismissals, settlements)**
- **Answer the question accurately, even with "negative" outcomes**

REMOVE cases with:
- Major contradictions between facts and reasoning
- Incorrect legal citations or misapplied laws
- Reasoning that doesn't address the stated legal question AT ALL
- Judgment that contradicts the reasoning
- Missing or nonsensical critical information
- Factual inconsistencies that undermine the case

IMPORTANT: A case that says "No, the court did not do X because Y happened" IS a valid answer to "Did the court do X?" Don't remove cases just because the outcome was procedural or negative.

Be thorough but not overly strict - keep cases that are legally sound even if the outcome wasn't a substantive ruling.
Respond with one decision per case, in the exact order provided.
"""

# ---------- HELPERS ----------
def load_json_file(path, default, logger):
    """Safely load JSON file with error handling"""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                count = len(data) if isinstance(data, (list, dict)) else 0
                logger.info(f"✅ Loaded {count} items from {path}")
                return data
        else:
            logger.info(f"📝 File {path} doesn't exist, using default")
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
        path.parent.mkdir(parents=True, exist_ok=True)
        
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        temp_path.replace(path)
        
        count = len(data) if isinstance(data, (list, dict)) else 0
        logger.debug(f"💾 Saved {count} items to {path}")
        
    except Exception as e:
        logger.error(f"❌ Error saving {path}: {e}")
        if temp_path.exists():
            temp_path.unlink()

def print_tracking_summary(tracking, logger):
    """Print detailed tracking statistics"""
    if not tracking:
        logger.info("📊 No tracking data available")
        return
        
    keep_count = sum(1 for v in tracking.values() if v.get("decision") == "KEEP")
    remove_count = sum(1 for v in tracking.values() if v.get("decision") == "REMOVE")
    fail_count = sum(1 for v in tracking.values() if v.get("status") == "fail")
    invalid_count = sum(1 for v in tracking.values() if v.get("status") == "invalid")
    total_tracked = len(tracking)
    
    logger.info("📊 Tracking Summary:")
    logger.info(f"   Total Tracked Cases: {total_tracked}")
    logger.info(f"   ✅ KEPT: {keep_count}")
    logger.info(f"   ❌ REMOVED: {remove_count}")
    logger.info(f"   🔄 Failed: {fail_count}")
    logger.info(f"   ⚠️ Invalid: {invalid_count}")
    
    if remove_count > 0:
        logger.info("   Removed Case Reasons:")
        removed_cases = [(cid, info) for cid, info in tracking.items() 
                        if info.get("decision") == "REMOVE"]
        for cid, info in removed_cases[:5]:
            logger.info(f"      {cid}: {info.get('reason', 'No reason provided')}")
        if len(removed_cases) > 5:
            logger.info(f"      ... and {len(removed_cases) - 5} more removed cases")

# ---------- THREAD-SAFE FILE MANAGER ----------
class FileManager:
    def __init__(self, logger):
        self.logger = logger
        self.lock = Lock()
    
    async def update_results_and_tracking(self, results_file, tracking_file, new_results, new_tracking):
        """Atomically update both results and tracking files"""
        async with self.lock:
            try:
                current_results = load_json_file(results_file, [], self.logger)
                current_tracking = load_json_file(tracking_file, {}, self.logger)
                
                if isinstance(current_results, list) and isinstance(new_results, list):
                    current_results.extend(new_results)
                
                if isinstance(current_tracking, dict) and isinstance(new_tracking, dict):
                    current_tracking.update(new_tracking)
                
                save_json_file(results_file, current_results, self.logger)
                save_json_file(tracking_file, current_tracking, self.logger)
                
                return current_results, current_tracking
                
            except Exception as e:
                self.logger.error(f"❌ Error updating files: {e}")
                return None, None

# ---------- API KEY MANAGER ----------
class APIKeyManager:
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
            
            available = [
                k for k in self.keys 
                if k["failures"] < RETRY_CYCLES 
                and k["cooldown_until"] <= current_time
                and k["consecutive_failures"] < 3
            ]
            
            if not available:
                available = [k for k in self.keys if k["cooldown_until"] <= current_time]
                
            if not available:
                available = [min(self.keys, key=lambda k: k["consecutive_failures"])]
                self.logger.warning("⚠️ All keys exhausted, using least failed key")
            
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
                    
                    if k["consecutive_failures"] >= 2:
                        cooldown_time = min(60, k["consecutive_failures"] * 10)
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
                    k["consecutive_failures"] = 0
                    k["cooldown_until"] = 0
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

# ---------- GEMINI API CALL ----------
async def call_gemini_verify(session, entries, api_manager, logger, max_retries=RETRY_CYCLES):
    """Call Gemini API for case verification"""
    
    def _truncate_text(text, max_len=3000):
        if not isinstance(text, str) or len(text) <= max_len:
            return text
        return text[:max_len] + " ...[truncated]"
    
    # Create slimmed version for API call
    slim_entries = []
    for case in entries:
        slim_case = dict(case)
        analysis = slim_case.get("analysis", {})
        
        case_summary = analysis.get("case_summary", {})
        for field in ("facts", "law_applied", "judgment"):
            if field in case_summary:
                case_summary[field] = _truncate_text(case_summary[field])
        
        court_reasoning = analysis.get("court_reasoning", {})
        for field in ("reasoning", "decision"):
            if field in court_reasoning:
                court_reasoning[field] = _truncate_text(court_reasoning[field])
        
        slim_entries.append(slim_case)

    prompt_text = f"{VERIFY_PROMPT}\n\nHere are the cases to verify:\n{json.dumps(slim_entries, ensure_ascii=False)}"
    logger.debug(f"📤 Payload size: {len(prompt_text)} characters for {len(entries)} cases")
    
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    headers = {"Content-Type": "application/json"}

    # Try each model variant
    for model_name in MODEL_VARIANTS:
        logger.debug(f"🤖 Trying model: {model_name}")
        
        for attempt in range(max_retries):
            api_key = await api_manager.get_key()
            
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                
                timeout = aiohttp.ClientTimeout(total=60)
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                    body_text = await resp.text()
                    
                    if resp.status != 200:
                        logger.warning(f"⚠️ HTTP {resp.status}: {body_text[:500]}")
                        
                        if resp.status == 429:
                            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                            logger.info(f"⏳ Rate limited, backing off {wait:.1f}s...")
                            await asyncio.sleep(wait)
                            await api_manager.record_failure(api_key)
                            continue
                            
                        elif 500 <= resp.status < 600:
                            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1)
                            logger.info(f"⏳ Server error, backing off {wait:.1f}s...")
                            await asyncio.sleep(wait)
                            await api_manager.record_failure(api_key)
                            continue
                            
                        else:
                            await api_manager.record_failure(api_key)
                            break
                    
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
                    
                    content = candidates[0].get("content", {})
                    if isinstance(content, list) and len(content) > 0:
                        content = content[0]
                    
                    parts = content.get("parts", [])
                    raw_text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    
                    if not raw_text:
                        logger.warning(f"⚠️ Empty response from {model_name}")
                        await api_manager.record_failure(api_key)
                        continue
                    
                    await api_manager.record_success(api_key)
                    logger.debug(f"✅ Successfully got response: {raw_text[:200].replace(chr(10), ' ')}")
                    return raw_text.strip()
                        
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

# ---------- RESPONSE PROCESSING ----------
def process_verification_response(input_cases, raw_response, logger):
    """Process simple text verification response"""
    kept_cases = []
    tracking_updates = {}
    
    try:
        # Split response into decision lines
        decisions = [line.strip() for line in raw_response.split('\n') if line.strip()]
        
        logger.debug(f"🔄 Processing {len(decisions)} decisions for {len(input_cases)} cases")
        
        # Match decisions to cases
        for i, case in enumerate(input_cases):
            case_id = case.get("case_id", f"case_{i}")
            
            if i < len(decisions):
                decision_line = decisions[i].upper()
                
                if "KEEP" in decision_line:
                    kept_cases.append(case)  # Add original case as-is
                    tracking_updates[case_id] = {
                        "status": "success",
                        "decision": "KEEP",
                        "processed_at": time.time()
                    }
                    logger.debug(f"✅ KEPT case {case_id}")
                    
                elif "REMOVE" in decision_line:
                    # Extract reason after "REMOVE:"
                    reason = "Verification failed"
                    if ":" in decisions[i]:
                        reason = decisions[i].split(":", 1)[1].strip()
                    
                    tracking_updates[case_id] = {
                        "status": "success", 
                        "decision": "REMOVE",
                        "reason": reason,
                        "processed_at": time.time()
                    }
                    logger.debug(f"❌ REMOVED case {case_id}: {reason}")
                    
                else:
                    # Unclear decision - default to keep with warning
                    kept_cases.append(case)
                    tracking_updates[case_id] = {
                        "status": "success",
                        "decision": "KEEP",
                        "reason": "Unclear decision, defaulted to KEEP",
                        "processed_at": time.time()
                    }
                    logger.warning(f"⚠️ Unclear decision for {case_id}: {decisions[i]}")
            else:
                # No decision available - default to keep
                kept_cases.append(case)
                tracking_updates[case_id] = {
                    "status": "success",
                    "decision": "KEEP", 
                    "reason": "No decision received, defaulted to KEEP",
                    "processed_at": time.time()
                }
                logger.warning(f"⚠️ No decision for {case_id}, defaulted to KEEP")
        
        return kept_cases, tracking_updates
        
    except Exception as e:
        logger.error(f"❌ Error processing verification response: {e}")
        logger.debug(f"🔍 Raw response: {raw_response}")
        
        # Fallback - keep all cases but mark as uncertain
        fallback_tracking = {}
        for case in input_cases:
            case_id = case.get("case_id")
            fallback_tracking[case_id] = {
                "status": "success",
                "decision": "KEEP",
                "reason": f"Processing error, defaulted to KEEP: {str(e)}",
                "processed_at": time.time()
            }
        
        return input_cases, fallback_tracking

# ---------- BATCH PROCESSING ----------
async def process_verification_batch(session, batch, file_manager, output_file, tracking_file, api_manager, logger):
    """Process a batch of cases for verification"""
    batch_ids = [case.get("case_id", "unknown") for case in batch]
    logger.info(f"🔥 Processing verification batch: {batch_ids}")

    start_time = time.perf_counter()
    
    # Try batch processing first
    response = await call_gemini_verify(session, batch, api_manager, logger)
    
    if response:
        # Process batch response
        kept_cases, batch_tracking = process_verification_response(batch, response, logger)
        
        logger.info(f"✅ Batch verification successful: {len(kept_cases)}/{len(batch)} cases kept")
        
    else:
        # Batch failed - process individually
        logger.warning(f"❌ Batch failed, processing {len(batch)} cases individually")
        
        kept_cases = []
        batch_tracking = {}
        
        for case in batch:
            case_id = case.get("case_id")
            individual_success = False
            
            for retry in range(MAX_INDIVIDUAL_RETRIES):
                try:
                    individual_response = await call_gemini_verify(session, [case], api_manager, logger, max_retries=2)
                    
                    if individual_response:
                        individual_kept, individual_tracking = process_verification_response([case], individual_response, logger)
                        
                        kept_cases.extend(individual_kept)
                        batch_tracking.update(individual_tracking)
                        
                        logger.info(f"✅ Individual case {case_id} processed successfully")
                        individual_success = True
                        break
                        
                except Exception as e:
                    logger.error(f"❌ Individual processing error for {case_id} (retry {retry+1}): {e}")
                    if retry < MAX_INDIVIDUAL_RETRIES - 1:
                        await asyncio.sleep(RATE_LIMIT_DELAY * (retry + 1))
            
            if not individual_success:
                # Keep case by default if all processing fails
                kept_cases.append(case)
                batch_tracking[case_id] = {
                    "status": "fail", 
                    "reason": "Individual processing failed after retries",
                    "processed_at": time.time()
                }
                logger.error(f"❌ Case {case_id} failed after {MAX_INDIVIDUAL_RETRIES} individual attempts")

    # Save results atomically
    await file_manager.update_results_and_tracking(output_file, tracking_file, kept_cases, batch_tracking)
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    success_count = sum(1 for tracking in batch_tracking.values() 
                       if tracking.get("status") == "success")
    kept_count = len(kept_cases)
    
    logger.info(f"⚡ Batch {batch_ids} completed in {elapsed:.2f}s ({success_count}/{len(batch)} processed, {kept_count} kept)")
    
    return elapsed

# ---------- MAIN PIPELINE ----------
async def verify_cases(mode):
    """Main verification pipeline"""
    
    # Select files based on mode
    input_file = INPUT_FILES[mode]
    output_file = OUTPUT_FILES[mode]
    tracking_file = TRACKING_FILES[mode]
    checkpoint_file = CHECKPOINT_FILES[mode]
    log_file = LOG_FILES[mode]
    
    logger = setup_logging(log_file)
    logger.info("🚀 Starting Legal Case Verification Pipeline")

    # Log processing configuration
    logger.info("📋 Processing Configuration:")
    logger.info(f"   - Mode: {mode}")
    logger.info(f"   - Input: {input_file}")
    logger.info(f"   - Output: {output_file}")
    logger.info(f"   - Tracking: {tracking_file}")
    logger.info(f"   - Batch Size: {BATCH_MAX_ITEMS} cases")
    logger.info(f"   - Max Payload: {BATCH_MAX_CHARS} chars")
    logger.info(f"   - Concurrency: {MIN_CONCURRENCY}-{MAX_CONCURRENCY} (adaptive)")
    logger.info(f"   - API Keys: {len(API_KEYS)}")

    # Load input data
    all_cases = load_json_file(input_file, [], logger)
    if not all_cases:
        logger.error("❌ No cases found in input file")
        return

    # Load tracking data
    tracking = load_json_file(tracking_file, {}, logger)
    
    # Initialize file manager
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
        decision = tracking.get(case_id, {}).get("decision", "")
        
        if status == "success" and decision in ["KEEP", "REMOVE"]:
            completed_cases += 1
        elif status == "fail":
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
    logger.info(f"   ✅ Completed: {completed_cases}")
    logger.info(f"   ❌ Failed (to retry): {len(failed_cases)}")
    logger.info(f"   🆕 New cases: {len(new_cases)}")
    logger.info(f"   📋 To process: {to_process_count}")

    if to_process_count == 0:
        logger.info("✅ All cases already processed!")
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
        
        with tqdm(total=to_process_count, desc="Verifying cases", unit="case") as pbar:
            
            while processed_cases < to_process_count:
                batch_start_idx = processed_cases
                concurrent_batches = []
                batch_case_counts = []
                
                # Create concurrent batches
                for _ in range(current_concurrency):
                    if batch_start_idx >= to_process_count:
                        break
                    
                    batch_end_idx = min(batch_start_idx + BATCH_MAX_ITEMS, to_process_count)
                    batch = cases_to_process[batch_start_idx:batch_end_idx]
                    
                    # Check if batch is too large (character-wise)
                    batch_json = json.dumps(batch, ensure_ascii=False)
                    if len(batch_json) > BATCH_MAX_CHARS and len(batch) > 1:
                        logger.info(f"📦 Splitting oversized batch ({len(batch_json)} chars)")
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

                # Process batches concurrently
                round_start_time = time.perf_counter()
                batch_tasks = []
                
                for i, batch in enumerate(concurrent_batches):
                    if i > 0:
                        await asyncio.sleep(RATE_LIMIT_DELAY)
                    
                    task = asyncio.create_task(
                        process_verification_batch(session, batch, file_manager, output_file, tracking_file, api_manager, logger)
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
                keep_count = sum(1 for v in current_tracking.values() if v.get("decision") == "KEEP")
                remove_count = sum(1 for v in current_tracking.values() if v.get("decision") == "REMOVE")
                fail_count = sum(1 for v in current_tracking.values() if v.get("status") == "fail")

                logger.info(f"Round {batch_index} completed in {round_duration:.2f}s")
                logger.info(f"Progress: {processed_cases}/{to_process_count} processed, {keep_count} kept, {remove_count} removed, {fail_count} failed")

                # Adaptive concurrency adjustment
                valid_durations = [d for d in batch_durations if isinstance(d, (int, float))]
                if valid_durations:
                    avg_batch_duration = sum(valid_durations) / len(valid_durations)
                    
                    if avg_batch_duration < TARGET_FAST and current_concurrency < MAX_CONCURRENCY:
                        current_concurrency += 1
                        logger.info(f"Increasing concurrency -> {current_concurrency}")
                    elif avg_batch_duration > TARGET_SLOW and current_concurrency > MIN_CONCURRENCY:
                        current_concurrency -= 1
                        logger.info(f"Decreasing concurrency -> {current_concurrency}")

                # Small delay between rounds
                if processed_cases < to_process_count:
                    await asyncio.sleep(RATE_LIMIT_DELAY)

    # Final statistics
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time

    # Load final tracking data
    final_tracking = load_json_file(tracking_file, {}, logger)
    final_keep = sum(1 for v in final_tracking.values() if v.get("decision") == "KEEP")
    final_remove = sum(1 for v in final_tracking.values() if v.get("decision") == "REMOVE")
    final_fail = sum(1 for v in final_tracking.values() if v.get("status") == "fail")
    final_completed = final_keep + final_remove

    # Load final output data for verification count
    final_output = load_json_file(output_file, [], logger)
    actual_kept_count = len(final_output)

    keep_rate = (final_keep / total_cases * 100) if total_cases > 0 else 0
    completion_rate = (final_completed / total_cases * 100) if total_cases > 0 else 0

    logger.info("\nFinal Verification Results:")
    logger.info(f"   Total input cases: {total_cases}")
    logger.info(f"   Cases processed this run: {to_process_count}")
    logger.info(f"   Total kept: {final_keep} ({keep_rate:.1f}%)")
    logger.info(f"   Total removed: {final_remove}")
    logger.info(f"   Total failed: {final_fail}")
    logger.info(f"   Total completed: {final_completed} ({completion_rate:.1f}%)")
    logger.info(f"   Cases in output file: {actual_kept_count}")
    logger.info(f"   Total pipeline time: {total_duration:.2f} seconds")
    
    if to_process_count > 0:
        avg_time_per_case = total_duration / to_process_count
        logger.info(f"   Average time per case: {avg_time_per_case:.2f} seconds")

    # Print detailed tracking summary
    print_tracking_summary(final_tracking, logger)
    
    # Print API key performance
    api_manager.print_stats()

    # Quality assessment
    if completion_rate >= 95:
        logger.info("Excellent! Over 95% of cases processed successfully")
    elif completion_rate >= 90:
        logger.info("Good! Over 90% of cases processed")
    elif completion_rate >= 80:
        logger.info(f"Acceptable, but {100 - completion_rate:.1f}% of cases still need attention")
    else:
        logger.warning("Many cases failed - please check API keys and input data quality")

    # Verification-specific assessment
    if final_keep > 0:
        quality_rate = (final_keep / (final_keep + final_remove) * 100) if (final_keep + final_remove) > 0 else 0
        logger.info(f"Verification Quality: {quality_rate:.1f}% of processed cases were kept")
    
    logger.info("Verification pipeline completed successfully!")

# ---------- GRACEFUL SHUTDOWN HANDLER ----------
async def graceful_shutdown(output_file, tracking_file, logger):
    """Handle graceful shutdown and save current state"""
    logger.info("Graceful shutdown initiated...")
    
    try:
        results = load_json_file(output_file, [], logger)
        tracking = load_json_file(tracking_file, {}, logger)
        
        save_json_file(output_file, results, logger)
        save_json_file(tracking_file, tracking, logger)
        
        logger.info("Current state saved successfully")
        print_tracking_summary(tracking, logger)
        
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    # Validate mode
    if MODE not in ["civil", "criminal"]:
        print("Invalid MODE. Must be 'civil' or 'criminal'")
        exit(1)

    try:
        asyncio.run(verify_cases(MODE))
        
    except KeyboardInterrupt:
        logger = setup_logging(LOG_FILES[MODE])
        logger.info("Processing interrupted by user")
        asyncio.run(graceful_shutdown(OUTPUT_FILES[MODE], TRACKING_FILES[MODE], logger))
        
    except Exception as e:
        logger = setup_logging(LOG_FILES[MODE])
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        asyncio.run(graceful_shutdown(OUTPUT_FILES[MODE], TRACKING_FILES[MODE], logger))