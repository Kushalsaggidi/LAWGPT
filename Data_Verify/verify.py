#!/usr/bin/env python3
"""
KGDG Verification Pipeline v4 - With Robust Tracking & Progress
Enhanced with comprehensive tracking, progress bars, and resume capability.
Modified to remove content length filtering, align with input data structure, and add logging, schema validation, and optimized batch processing.
Designed to remove invalid cases without correcting them.
"""

import asyncio
import argparse
import json
import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from jsonschema import validate, ValidationError

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("verify.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
def parse_args():
    """Parse command-line arguments for file paths."""
    parser = argparse.ArgumentParser(description="KGDG Verification Pipeline")
    parser.add_argument("--input-dir", default="Data_Polish", help="Directory for input JSON files")
    parser.add_argument("--output-dir", default="Data_Verify", help="Directory for output and tracking JSON files")
    parser.add_argument("--mode", default="civil", choices=["civil", "criminal", "both"], help="Processing mode")
    return parser.parse_args()

args = parse_args()
MODE = args.mode

INPUT_FILES = {
    "civil": r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_civil.json",
    "criminal": r"C:\Users\Kushal\Desktop\new\Data_Polish\polish_criminal.json"
}

OUTPUT_FILES = {
    "civil": r"C:\Users\Kushal\Desktop\new\Data_Verify\verified_civil.json",
    "criminal": r"C:\Users\Kushal\Desktop\new\Data_Verify\verified_criminal.json"
}

TRACKING_FILES = {
    "civil": r"C:\Users\Kushal\Desktop\new\Data_Verify\tracking_civil.json",
    "criminal": r"C:\Users\Kushal\Desktop\new\Data_Verify\tracking_criminal.json"
}

# API Configuration
API_KEYS = [
    'AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM',
    'AIzaSyCz41qZibYl1rca9Ye7dRV9dK5Hf-ZzbA',
    'AIzaSyA27TjinZixsBBl1ZkY7dRV9dK5Hf-ZzbA',
    'AIzaSyBFEIHo4GKXri8a3rNAaQKq0EwRSrwjwYs',
    'AIzaSyCAEEIT9lc5WxLAyq_N4SaKouAggntDxC4'
]

MODEL_TRY = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]

# Settings
CONCURRENCY = 4
BATCH_SIZE = 3
MAX_RETRIES = 3
RETRY_DELAY = 2

# Quality control
CONFIDENCE_THRESHOLD = 0.6

# Retryable failure reasons
RETRYABLE_REASONS = [
    "api_quota_exceeded",
    "rate_limit_hit",
    "timeout_error",
    "network_error",
    "model_unavailable",
    "generic_api_error"
]

# Input schema for validation
CASE_SCHEMA = {
    "type": "object",
    "required": ["case_id", "instruction", "question", "analysis"],
    "properties": {
        "case_id": {"type": "string"},
        "instruction": {"type": "string"},
        "question": {"type": "string"},
        "analysis": {
            "type": "object",
            "required": ["case_summary"],
            "properties": {
                "case_summary": {
                    "type": "object",
                    "required": ["reasoning", "law_applied"],
                    "properties": {
                        "reasoning": {"type": "string"},
                        "law_applied": {"type": "string"},
                        "facts": {"type": "string"},
                        "legal_issue": {"type": "string"},
                        "judgment": {"type": "string"}
                    }
                },
                "court_reasoning": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "decision": {"type": "string"}
                    }
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "case_type": {"type": "string"},
                "law": {"type": "string"}
            }
        }
    }
}

# Verification prompts
QUICK_FILTER_PROMPT = """
TASK: Filter legal cases. Mark as false if ANY critical issue exists:

REJECT (false) if:
1. Missing required fields (instruction, question, case_summary.reasoning, case_summary.law_applied)
2. Placeholder text like "[placeholder]", "TODO", "example"
3. Gibberish or corrupted text
4. Duplicate content across fields
5. Non-English content
6. Empty or null fields

KEEP (true) if:
- Complete fields with real content
- Coherent legal language
- Proper structure

Return JSON array: [true, false, true, ...]
When uncertain, KEEP the case (true).
"""

CONFIDENCE_PROMPT = """
TASK: Score legal case quality with confidence.

Rate each case 0-1:
- 1.0 = Excellent, definitely keep
- 0.8+ = Good, likely keep
- 0.6+ = Acceptable, borderline
- 0.4+ = Poor, likely reject
- 0.0-0.3 = Seriously flawed

Return JSON array:
[
  {"keep": true, "confidence": 0.85, "reason": "Strong reasoning"},
  {"keep": false, "confidence": 0.2, "reason": "Poor quality"},
  ...
]
"""

# ==================== UTILITIES ====================
def load_json(file_path: str) -> Any:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {} if 'tracking' in file_path else []

def save_json_atomic(data: Any, file_path: str) -> None:
    """Save JSON file atomically to prevent corruption."""
    temp_path = f"{file_path}.tmp"
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, file_path)
    except Exception as e:
        logger.error(f"Error saving {file_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_case_id(case: Dict) -> str:
    """Get case identifier."""
    return case.get("id", case.get("case_id", f"case_{hash(str(case)) % 100000}"))

def clean_response(text: str) -> str:
    """Clean API response to extract JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    
    array_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if array_match:
        return array_match.group(0)
    
    logger.error(f"Malformed API response: {text[:100]}...")
    raise ValueError("Could not extract valid JSON array")

def calculate_content_length(case: Dict) -> int:
    """Calculate total content length based on actual case structure."""
    parts = []

    # Include instruction and question
    parts.append(str(case.get("instruction", "")))
    parts.append(str(case.get("question", "")))

    # Handle analysis block
    analysis = case.get("analysis", {})
    if isinstance(analysis, dict):
        for section in analysis.values():
            if isinstance(section, dict):
                parts.extend(str(v) for v in section.values())
            elif isinstance(section, str):
                parts.append(section)

    # Include metadata
    metadata = case.get("metadata", {})
    if isinstance(metadata, dict):
        parts.extend(str(v) for v in metadata.values())

    content = " ".join(parts)
    return len(content.strip())

def has_basic_issues(case: Dict) -> bool:
    """Check for obvious structural issues."""
    # Check required fields
    if not case.get("instruction") or not case.get("question"):
        return True
    
    analysis = case.get("analysis", {})
    if not isinstance(analysis, dict):
        return True
    
    case_summary = analysis.get("case_summary", {})
    if not case_summary.get("reasoning") or not case_summary.get("law_applied"):
        return True
    
    # Check for placeholders
    all_text = " ".join([
        str(case.get("instruction", "")),
        str(case.get("question", "")),
        str(case_summary.get("reasoning", "")),
        str(case_summary.get("law_applied", "")),
        str(analysis.get("court_reasoning", {}).get("reasoning", ""))
    ]).lower()
    
    placeholders = ["[placeholder]", "todo", "example", "sample", "insert", "xxx"]
    return any(p in all_text for p in placeholders)

def format_time(seconds: float) -> str:
    """Format time duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def print_progress_bar(current: int, total: int, width: int = 40) -> str:
    """Create a simple progress bar."""
    if total == 0:
        return "[" + "=" * width + "] 100%"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}%"

# ==================== API MANAGER ====================
class APIManager:
    def __init__(self):
        self.keys = API_KEYS.copy()
        self.models = MODEL_TRY.copy()
        self.current_key = 0
        self.current_model = 0
        self.exhausted_keys = set()
    
    def get_current_key(self) -> str:
        available = [k for i, k in enumerate(self.keys) if i not in self.exhausted_keys]
        if not available:
            self.exhausted_keys.clear()  # Reset
            available = self.keys
        return available[self.current_key % len(available)]
    
    def get_current_model(self) -> str:
        return self.models[self.current_model % len(self.models)]
    
    def rotate_key(self):
        self.current_key += 1
    
    def rotate_model(self):
        self.current_model += 1
    
    def mark_key_exhausted(self, key: str):
        try:
            idx = self.keys.index(key)
            self.exhausted_keys.add(idx)
        except ValueError:
            pass

# ==================== TRACKING MANAGER ====================
class TrackingManager:
    def __init__(self, tracking_file: str):
        self.tracking_file = tracking_file
        self.data = self._load_tracking()
    
    def _load_tracking(self) -> Dict:
        """Load tracking data with metadata."""
        data = load_json(self.tracking_file)
        
        if not isinstance(data, dict):
            data = {}
        
        if "_meta" not in data:
            data["_meta"] = {
                "total_cases": 0,
                "successful": 0,
                "failed_permanent": 0,
                "failed_retryable": 0,
                "last_updated": datetime.now().isoformat(),
                "start_time": datetime.now().isoformat()
            }
        
        return data
    
    def save_tracking(self):
        """Save tracking data atomically."""
        self.data["_meta"]["last_updated"] = datetime.now().isoformat()
        save_json_atomic(self.data, self.tracking_file)
    
    def should_process_case(self, case_id: str) -> bool:
        """Check if case should be processed."""
        if case_id not in self.data:
            return True
        
        entry = self.data[case_id]
        if entry["status"] == "success":
            return False
        
        reason = entry.get("reason", "")
        return reason in RETRYABLE_REASONS
    
    def mark_case_success(self, case_id: str):
        """Mark case as successful."""
        if case_id in self.data:
            old_entry = self.data[case_id]
            if old_entry["status"] == "fail":
                reason = old_entry.get("reason", "")
                if reason in RETRYABLE_REASONS:
                    self.data["_meta"]["failed_retryable"] -= 1
                else:
                    self.data["_meta"]["failed_permanent"] -= 1
        else:
            self.data["_meta"]["total_cases"] += 1
        
        self.data[case_id] = {"status": "success"}
        self.data["_meta"]["successful"] += 1
    
    def mark_case_failed(self, case_id: str, reason: str):
        """Mark case as failed with reason."""
        if case_id in self.data:
            old_entry = self.data[case_id]
            if old_entry["status"] == "success":
                self.data["_meta"]["successful"] -= 1
        else:
            self.data["_meta"]["total_cases"] += 1
        
        self.data[case_id] = {
            "status": "fail",
            "reason": reason
        }
        
        if reason in RETRYABLE_REASONS:
            self.data["_meta"]["failed_retryable"] += 1
        else:
            self.data["_meta"]["failed_permanent"] += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.data["_meta"].copy()
    
    def print_tracking_summary(self):
        """Print detailed tracking statistics."""
        meta = self.data["_meta"]
        total = meta["total_cases"]
        successful = meta["successful"]
        failed_perm = meta["failed_permanent"]
        failed_retry = meta["failed_retryable"]
        
        logger.info("TRACKING FILE STATUS:")
        logger.info(f"Total tracked cases: {total}")
        logger.info(f"Successful: {successful} ({successful/total*100 if total > 0 else 0:.1f}%)")
        logger.info(f"Failed (permanent): {failed_perm} ({failed_perm/total*100 if total > 0 else 0:.1f}%)")
        logger.info(f"Failed (retryable): {failed_retry} ({failed_retry/total*100 if total > 0 else 0:.1f}%)")
        
        failure_reasons = {}
        for case_id, entry in self.data.items():
            if case_id.startswith("_"):
                continue
            if entry.get("status") == "fail":
                reason = entry.get("reason", "unknown")
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        if failure_reasons:
            logger.info("Top failure reasons:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  • {reason}: {count}")

# ==================== VERIFIER ====================
class CaseVerifier:
    def __init__(self, case_type: str):
        self.case_type = case_type
        self.api_manager = APIManager()
        self.tracking = TrackingManager(TRACKING_FILES[case_type])
        self.verified_cases = []
        self.batch_count = 0
        self.start_time = time.time()
        
        self.stats = {
            "processed_this_run": 0,
            "successful_this_run": 0,
            "failed_this_run": 0,
            "batches_completed": 0,
            "api_errors": 0
        }
    
    def basic_filter(self, cases: List[Dict]) -> List[Dict]:
        """Filter cases with obvious structural issues."""
        logger.info("Basic filtering...")
        filtered = []
        
        for case in cases:
            case_id = get_case_id(case)
            
            try:
                validate(instance=case, schema=CASE_SCHEMA)
            except ValidationError as e:
                logger.warning(f"Case {case_id} failed schema validation: {e.message}")
                self.tracking.mark_case_failed(case_id, "invalid_data_structure")
                continue
            
            if has_basic_issues(case):
                self.tracking.mark_case_failed(case_id, "invalid_data_structure")
                continue
            
            filtered.append(case)
        
        removed = len(cases) - len(filtered)
        if removed > 0:
            logger.info(f"Basic filter removed: {removed} cases")
            self.tracking.save_tracking()
        
        return filtered
    
    async def verify_batch_api(self, cases: List[Dict], prompt: str) -> List[Dict]:
        """Make API call to verify batch."""
        for attempt in range(MAX_RETRIES + 1):
            current_key = self.api_manager.get_current_key()
            current_model = self.api_manager.get_current_model()
            
            try:
                genai.configure(api_key=current_key)
                model = genai.GenerativeModel(current_model)
                
                full_prompt = f"{prompt}\n\nCases:\n{json.dumps(cases, ensure_ascii=False, indent=1)}"
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=1000
                    )
                )
                
                result_text = clean_response(response.text)
                
                try:
                    decisions = json.loads(result_text)
                    
                    if decisions and isinstance(decisions[0], dict):
                        return [
                            {
                                "case": case,
                                "keep": decision.get("keep", True),
                                "confidence": decision.get("confidence", 1.0),
                                "reason": decision.get("reason", "")
                            }
                            for case, decision in zip(cases, decisions)
                        ]
                    else:
                        return [
                            {
                                "case": case,
                                "keep": keep,
                                "confidence": 1.0,
                                "reason": ""
                            }
                            for case, keep in zip(cases, decisions)
                        ]
                
                except json.JSONDecodeError:
                    bools = re.findall(r'(true|false)', result_text.lower())
                    if len(bools) == len(cases):
                        return [
                            {
                                "case": case,
                                "keep": b == 'true',
                                "confidence": 1.0,
                                "reason": ""
                            }
                            for case, b in zip(cases, bools)
                        ]
                    else:
                        raise ValueError(f"Could not parse response")
            
            except Exception as e:
                error_str = str(e)
                self.stats["api_errors"] += 1
                
                if "quota" in error_str.lower() or "403" in error_str:
                    self.api_manager.mark_key_exhausted(current_key)
                    self.api_manager.rotate_key()
                    return [{"error": "api_quota_exceeded", "case": case} for case in cases]
                
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(min(wait_time, 30))
                    continue
                
                if attempt < MAX_RETRIES:
                    if attempt % 2 == 0:
                        self.api_manager.rotate_key()
                    else:
                        self.api_manager.rotate_model()
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                
                return [{"error": "generic_api_error", "case": case} for case in cases]
        
        return [{"error": "generic_api_error", "case": case} for case in cases]
    
    async def process_batch(self, cases: List[Dict], prompt: str, stage_name: str) -> List[Dict]:
        """Process a single batch with progress tracking."""
        batch_start = time.time()
        batch_size = len(cases)
        
        logger.info(f"Processing batch of {batch_size} cases...")
        
        decisions = await self.verify_batch_api(cases, prompt)
        
        successful_cases = []
        failed_cases = []
        
        for decision in decisions:
            if "error" in decision:
                failed_cases.append(decision["case"])
            else:
                case = decision["case"]
                case_id = get_case_id(case)
                keep = decision["keep"]
                confidence = decision.get("confidence", 1.0)
                
                if confidence < CONFIDENCE_THRESHOLD:
                    keep = False
                
                if keep:
                    successful_cases.append(case)
                    self.tracking.mark_case_success(case_id)
                    self.stats["successful_this_run"] += 1
                else:
                    self.tracking.mark_case_failed(case_id, "failed_quality_check")
                    self.stats["failed_this_run"] += 1
                self.stats["processed_this_run"] += 1
        
        if failed_cases:
            logger.warning(f"{len(failed_cases)} cases failed in batch, retrying individually...")
            for case in failed_cases:
                individual_decision = (await self.verify_batch_api([case], prompt))[0]
                case_id = get_case_id(case)
                if "error" in individual_decision:
                    self.tracking.mark_case_failed(case_id, individual_decision["error"])
                    self.stats["failed_this_run"] += 1
                else:
                    keep = individual_decision["keep"]
                    confidence = individual_decision.get("confidence", 1.0)
                    if confidence < CONFIDENCE_THRESHOLD:
                        keep = False
                    if keep:
                        successful_cases.append(case)
                        self.tracking.mark_case_success(case_id)
                        self.stats["successful_this_run"] += 1
                    else:
                        self.tracking.mark_case_failed(case_id, "failed_quality_check")
                        self.stats["failed_this_run"] += 1
                    self.stats["processed_this_run"] += 1
        
        self.tracking.save_tracking()
        self._save_verified_cases()
        
        batch_time = time.time() - batch_start
        self.batch_count += 1
        self.stats["batches_completed"] += 1
        
        kept_count = len(successful_cases)
        total_time = time.time() - self.start_time
        
        logger.info(f"Batch complete: {kept_count}/{batch_size} kept")
        logger.info(f"Batch time: {format_time(batch_time)} | Total time: {format_time(total_time)}")
        
        return successful_cases
    
    def _save_verified_cases(self):
        """Save verified cases incrementally."""
        if self.verified_cases:
            output_file = OUTPUT_FILES[self.case_type]
            save_json_atomic(self.verified_cases, output_file)
    
    async def process_stage(self, cases: List[Dict], prompt: str, stage_name: str) -> List[Dict]:
        """Process a verification stage with progress tracking."""
        if not cases:
            return cases
        
        total_cases = len(cases)
        logger.info(f"{stage_name.upper()}: {total_cases} cases")
        
        batches = [cases[i:i + BATCH_SIZE] for i in range(0, len(cases), BATCH_SIZE)]
        total_batches = len(batches)
        
        verified_cases = []
        semaphore = asyncio.Semaphore(CONCURRENCY)
        processed_batches = 0
        
        async def process_single_batch(batch_cases):
            nonlocal processed_batches
            async with semaphore:
                batch_result = await self.process_batch(batch_cases, prompt, stage_name)
                processed_batches += 1
                
                progress_bar = print_progress_bar(processed_batches, total_batches)
                logger.info(f"Progress: {progress_bar} ({processed_batches}/{total_batches} batches)")
                
                return batch_result
        
        results = await asyncio.gather(*[process_single_batch(batch) for batch in batches])
        
        for batch_result in results:
            verified_cases.extend(batch_result)
            self.verified_cases.extend(batch_result)
        
        removed = total_cases - len(verified_cases)
        logger.info(f"{stage_name} complete: {len(verified_cases)}/{total_cases} kept ({removed} removed)")
        
        return verified_cases
    
    def print_processing_summary(self, cases_to_process: List[Dict], new_cases: List[Dict], retry_cases: List[Dict]):
        """Print processing plan summary."""
        logger.info("PROCESSING PLAN:")
        logger.info(f"Total input cases: {len(cases_to_process)}")
        logger.info(f"New cases: {len(new_cases)}")
        logger.info(f"Retry cases: {len(retry_cases)}")
        logger.info(f"Batch size: {BATCH_SIZE} | Concurrency: {CONCURRENCY}")
    
    async def process_file(self):
        """Process the verification pipeline."""
        input_file = INPUT_FILES[self.case_type]
        
        logger.info(f"Processing {self.case_type.upper()} cases from {input_file}")
        
        all_cases = load_json(input_file)
        if not all_cases:
            logger.error(f"No cases found in {input_file}")
            return
        
        logger.info(f"Loaded {len(all_cases)} total cases")
        
        self.tracking.print_tracking_summary()
        
        new_cases = []
        retry_cases = []
        skipped_count = 0
        
        for case in all_cases:
            case_id = get_case_id(case)
            
            if not self.tracking.should_process_case(case_id):
                skipped_count += 1
                if case_id in self.tracking.data and self.tracking.data[case_id]["status"] == "success":
                    self.verified_cases.append(case)
            else:
                if case_id in self.tracking.data:
                    retry_cases.append(case)
                else:
                    new_cases.append(case)
        
        cases_to_process = new_cases + retry_cases
        
        if not cases_to_process:
            logger.info("All cases already processed successfully!")
            self._save_verified_cases()
            return
        
        logger.info(f"Skipped {skipped_count} already successful cases")
        self.print_processing_summary(cases_to_process, new_cases, retry_cases)
        
        current_cases = self.basic_filter(cases_to_process)
        
        if not current_cases:
            logger.info("No cases remaining after basic filtering")
            self._save_verified_cases()
            return
        
        logger.info("Starting verification pipeline...")
        
        current_cases = await self.process_stage(current_cases, QUICK_FILTER_PROMPT, "Quick Filter")
        
        if current_cases:
            current_cases = await self.process_stage(current_cases, CONFIDENCE_PROMPT, "Detailed Filter")
        
        self._save_verified_cases()
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print comprehensive final summary."""
        total_time = time.time() - self.start_time
        tracking_stats = self.tracking.get_stats()
        
        logger.info(f"{self.case_type.upper()} VERIFICATION COMPLETE!")
        logger.info(f"Total time: {format_time(total_time)}")
        logger.info(f"Batches processed: {self.stats['batches_completed']}")
        
        logger.info("THIS RUN STATS:")
        logger.info(f"Processed: {self.stats['processed_this_run']} cases")
        logger.info(f"Successful: {self.stats['successful_this_run']}")
        logger.info(f"Failed: {self.stats['failed_this_run']}")
        
        if self.stats["api_errors"] > 0:
            logger.info(f"API errors: {self.stats['api_errors']}")
        
        logger.info("OVERALL PROGRESS:")
        total = tracking_stats["total_cases"]
        successful = tracking_stats["successful"]
        logger.info(f"Total cases: {total}")
        logger.info(f"Total successful: {successful} ({successful/total*100 if total > 0 else 0:.1f}%)")
        logger.info(f"Output file: {OUTPUT_FILES[self.case_type]}")
        logger.info(f"Tracking file: {TRACKING_FILES[self.case_type]}")
        
        if total_time > 0:
            cases_per_second = self.stats["processed_this_run"] / total_time
            logger.info(f"Processing speed: {cases_per_second:.2f} cases/second")

# ==================== MAIN ====================
async def main():
    """Main function."""
    logger.info("KGDG Verification Pipeline v4 - With Tracking & Progress")
    logger.info(f"Mode: {MODE}")
    logger.info(f"API Keys: {len(API_KEYS)} available")
    logger.info(f"Models: {', '.join(MODEL_TRY)}")
    logger.info("-" * 60)
    
    if MODE.lower() == "both":
        file_types = ["civil", "criminal"]
    elif MODE.lower() in ["civil", "criminal"]:
        file_types = [MODE.lower()]
    else:
        logger.error(f"Invalid MODE: {MODE}")
        return
    
    for file_type in file_types:
        verifier = CaseVerifier(file_type)
        await verifier.process_file()
        logger.info("-" * 60)
    
    logger.info("All verification tasks completed!")

if __name__ == "__main__":
    asyncio.run(main())