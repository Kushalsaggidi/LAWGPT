#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KGDG – verify.py (Windows-safe, async, resumable, index-safe, with auto-repair)

What this script does (high level):
- Loads cases from cases_polished.json (order preserved, length N)
- Builds batches (size- and char-budget aware)
- Runs verification calls to Gemini with bounded concurrency
- Uses a single saver task to write files every few seconds (no Windows file lock collisions)
- Keeps indexing stable: verified output is a list of length N (each slot is either the original case or a fail marker)
- Writes a failed-only JSON with reasons for quick triage
- Automatically re-runs only the failed cases for several *cycles* (up to MAX_PASSES)
- Tries targeted *single-case* retries when batches fail due to JSON formatting, rate-limits (429/quota), or 500 errors

Console messages you’ll see:
- “Loaded … cases”
- “To process this run: X / N cases”
- “🚀 First pass: …”
- “🧷 Saver started …”
- “✅ Batch done …”
- periodic “💾 Saved: …”
- “🔁 Retry: … cases …”
- Final summary: success/fail counts and file locations
"""

from __future__ import annotations
import asyncio
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Tuple, Optional

import google.generativeai as genai

# ====================== CONFIG (kept same names) ======================

INPUT_FILE = "cases_polished.json"                 # Input; order & length preserved
VERIFIED_FILE = "cases_verified.json"              # All slots present; failed slots have fail object
FAILED_FILE = "verification_failed_cases.json"     # Only failures (original + reason)
TRACKING_FILE = "verify_tracking.json"             # Pass/fail per index; used for resuming

# Keep your API key handling as-is. You can hardcode or export as env var.
GOOGLE_API_KEY = "AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM"
GEMINI_MODEL = "gemini-2.5-flash-lite"             # Model choice unchanged

# Concurrency & batching
MAX_CONCURRENCY = 5               # <= You asked 4-5; set to 5. Controls simultaneous in-flight requests
BATCH_SIZE = 5                    # outer batch size (unchanged)
MAX_CHARS_PER_REQUEST = 14000     # char budget per request (unchanged)

# Retry handling (for each API call)
MAX_RETRIES = 3                   # batch call retries
BASE_BACKOFF = 2.0                # seconds
BACKOFF_JITTER = (0.5, 1.5)       # multiplicative jitter

# Saver (single writer) interval
SAVE_INTERVAL_SECONDS = 5.0       # flush to disk every 5s

# Multi-cycle verification
MAX_PASSES = 5                    # <= NEW: run up to 5 cycles automatically until fails settle

# Version tag for auditing
VERIFY_VERSION = "kgdg-verify-v3.2-2025-08-14"

# ====================== PROMPT (verify-only) ======================

VERIFY_PROMPT = """
You are given an array of JSON legal case objects.

Your job:
1. For each case, verify correctness and completeness:
   - All fields ("instruction", "question", "answer.reasoning", and "answer.reference") must be non-empty, specific, and meaningful — not placeholders or generic text.
   - Facts in "reasoning" must align with the "question" and be logically consistent.
   - Legal references must be factually correct, relevant to the charge, and consistent with the reasoning.
   - No contradictory information should be present.
2. If a case is fully correct, return it EXACTLY as given without modifying wording, formatting, or order.
3. If a case fails any check, return ONLY:
   {"status": "fail", "reason": "<clear, concise reason for failure>"} 
   in place of that case.
4. Do not attempt to fill missing fields or rewrite stylistic elements — only verify.
5. Output ONLY a valid JSON array matching the input order, with either the original case or a failure object in each position.
"""

# ====================== UTILS ======================

def load_json_safe(path: str, default: Any) -> Any:
    """Safely load JSON file; return default on missing/invalid content."""
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error reading {path}. Using default.")
        return default

def save_json_atomic(data: Any, path: str) -> None:
    """Atomic save to avoid partial writes on Windows."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)

def clean_code_fences(text: str) -> str:
    """Strip ``` fences and try to extract a JSON array if wrapped."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"(\[.*\])", t, flags=re.DOTALL)
    return m.group(1).strip() if m else t

def case_str_len(case: Dict[str, Any]) -> int:
    """Length of serialized JSON case (char budget helper)."""
    return len(json.dumps(case, ensure_ascii=False))

def dynamic_pack(
    keyed_cases: List[Tuple[int, Dict[str, Any]]],
    max_items: int,
    max_chars: int
) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """
    Pack (index, case) pairs into batches that respect:
      - max_items per batch
      - max_chars total serialized size per request
    """
    batches: List[List[Tuple[int, Dict[str, Any]]]] = []
    cur: List[Tuple[int, Dict[str, Any]]] = []
    cur_chars = 0
    for pair in keyed_cases:
        s = case_str_len(pair[1])
        if cur and (len(cur) >= max_items or (cur_chars + s) > max_chars):
            batches.append(cur)
            cur = [pair]
            cur_chars = s
        else:
            cur.append(pair)
            cur_chars += s
    if cur:
        batches.append(cur)
    return batches

def normalize_case_shape(obj: Any) -> Any:
    """
    Lenient normalization for harmless structural variations.
    Only touch benign issues; DO NOT rewrite semantics or style.
    """
    if not isinstance(obj, dict):
        return obj
    if obj.get("status") == "fail":
        return obj
    ans = obj.get("answer")
    if isinstance(ans, dict):
        reasoning = ans.get("reasoning")
        # Sometimes the model returns ["text"] instead of "text"
        if isinstance(reasoning, list) and len(reasoning) == 1 and isinstance(reasoning[0], str):
            ans["reasoning"] = reasoning[0]
    return obj

def is_fail_object(obj: Any) -> bool:
    """
    Return True only for *unrepaired* failure objects.
    Anything marked repaired (or carrying a fixed_case) should NOT be counted as a failure.
    """
    if not isinstance(obj, dict):
        return False
    if obj.get("status") != "fail":
        return False
    # If it's been repaired, don't treat it as a failure
    return not (obj.get("repaired") is True or obj.get("_repaired") is True or ("fixed_case" in obj))


def structured_fail(reason: str, error_type: Optional[str] = None) -> Dict[str, Any]:
    """Standard fail object."""
    out = {"status": "fail", "reason": reason}
    if error_type:
        out["error_type"] = error_type
    return out

def backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter."""
    mult = random.uniform(*BACKOFF_JITTER)
    delay = (BASE_BACKOFF * (2 ** (attempt - 1))) * mult
    time.sleep(delay)

def parse_retry_delay_seconds(err_text: str, default_wait: int = 15) -> int:
    """
    Parse 'retry_delay { seconds: N }' from Gemini error text when present,
    else return default_wait.
    """
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_text)
    if m:
        try:
            secs = int(m.group(1))
            return max(secs, default_wait)
        except Exception:
            return default_wait
    return default_wait

def is_retryable_exception_text(err: str) -> bool:
    """Classify retryable errors by substring (429/quota/timeout/temp/500/unavailable)."""
    e = err.lower()
    return any(k in e for k in ("429", "quota", "rate", "timeout", "temporar", "unavailable", "500", "internal error"))

def ensure_genai_configured():
    """Configure the Gemini client once."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("GOOGLE_API_KEY not set. Provide a valid key or set the env var GOOGLE_API_KEY.")
    genai.configure(api_key=GOOGLE_API_KEY)

# ====================== GEMINI CALLS ======================

def call_gemini_verify_sync(cases: List[Dict[str, Any]]) -> List[Any]:
    """
    Synchronous Gemini call. Returns a per-item list:
      - original case dict (possibly normalized) if correct, OR
      - fail object {"status":"fail","reason":"...","error_type": "..."} otherwise
    """
    payload = f"{VERIFY_PROMPT}\n\nHere is the JSON:\n{json.dumps(cases, ensure_ascii=False)}"
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        resp = model.generate_content(payload)
    except Exception as e:
        # Hard exception (e.g., network, quota). Mark the whole batch as exception.
        return [structured_fail(f"Verifier error: {str(e)}", "exception") for _ in cases]

    text = clean_code_fences(getattr(resp, "text", "") or "").strip()
    if not text:
        return [structured_fail("Empty response from model", "empty_output") for _ in cases]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [structured_fail("Invalid JSON output from model", "invalid_json") for _ in cases]

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list) or len(parsed) != len(cases):
        return [structured_fail("Verifier returned wrong-length or non-list JSON", "length_mismatch") for _ in cases]

    parsed = [normalize_case_shape(x) for x in parsed]
    return parsed

async def call_gemini_verify(cases: List[Dict[str, Any]]) -> List[Any]:
    """
    Async wrapper around the sync call with retries.
    - Retries the whole batch if a hard exception bubbles up
    - Uses exponential backoff and honors retry_delay when present in the error text
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(call_gemini_verify_sync, cases)
        except Exception as e:
            # This path is for unexpected exceptions thrown around the sync function
            err = str(e)
            if attempt < MAX_RETRIES and is_retryable_exception_text(err):
                # If server provided a retry_delay, respect it; otherwise exponential backoff
                wait_s = parse_retry_delay_seconds(err, default_wait=int(BASE_BACKOFF * attempt) + 2)
                time.sleep(wait_s)
                continue
            # Out of retries → mark each item as exception
            return [structured_fail(f"Verifier error: {err}", "exception") for _ in cases]

# ====================== SAVER QUEUE (single writer) ======================

class Saver:
    """
    Central saver (single writer) that receives results from workers via an asyncio.Queue
    and flushes to disk every SAVE_INTERVAL_SECONDS to avoid Windows file lock issues.

    - verified_cases: list[Any] of length N (None until filled)
    - tracking: dict[str, Any] with status per index (1-based keys)
    - failed_list: list[dict] with {"original_index", "reason", "error_type", "case"}
    """
    def __init__(
        self,
        verified_cases: List[Any],
        tracking: Dict[str, Any],
        failed_list: List[Dict[str, Any]],
        cases_source: List[Dict[str, Any]],
    ):
        self.verified_cases = verified_cases
        self.tracking = tracking
        self.failed_list = failed_list
        self.cases_source = cases_source
        self.queue: asyncio.Queue = asyncio.Queue()
        self._stop = asyncio.Event()
        self._last_flush = time.time()

    def _remove_failed_entry(self, index1: int):
        """Remove any stale failure record for this 1-based index (when we later succeed/repair)."""
        self.failed_list[:] = [f for f in self.failed_list if f.get("original_index") != index1]


    def _apply_results(self, indices: List[int], results: List[Any]):
        """Apply a batch of results into in-memory structures (index-safe)."""
        for idx, res in zip(indices, results):
            existing = self.verified_cases[idx]

            # ✅ Only skip if we already have a SUCCESS.
            # (If it's None or a previous FAIL, allow overwriting with a repair/success.)
            if existing is not None and not is_fail_object(existing):
                continue

            # ✅ If the worker says it "repaired" the case, treat it as SUCCESS immediately.
            if isinstance(res, dict) and (res.get("_repaired") or res.get("repaired") or ("fixed_case" in res)):
                fixed = res.get("fixed_case", res)

                # Normalize: strip any error-looking fields that could keep it classified as "fail"
                if isinstance(fixed, dict):
                    # Remove stray failure keys if they leaked through
                    if fixed.get("status") == "fail":
                        fixed.pop("status", None)
                    fixed.pop("error_type", None)
                    fixed.pop("reason", None)

                # Write repaired case as the SUCCESS record
                self.verified_cases[idx] = fixed
                self.tracking[str(idx + 1)] = {
                    "status": "success",
                    "verify": "Correct",
                    "message": "Case verified after auto-repair.",
                }
                # Remove any stale failure entry for this index
                self._remove_failed_entry(idx + 1)
                continue

            # ❌ Unrepaired failure → record as fail
            if is_fail_object(res):
                self.verified_cases[idx] = res
                # Keep only the latest fail record for this index
                self._remove_failed_entry(idx + 1)
                self.failed_list.append({
                    "original_index": idx + 1,  # 1-based for readability
                    "reason": res.get("reason", "Unknown"),
                    "error_type": res.get("error_type"),
                    "case": self.cases_source[idx],
                })
                self.tracking[str(idx + 1)] = {
                    "status": "fail",
                    "verify": "Incorrect",
                    "reason": res.get("reason", "Unknown"),
                    "error_type": res.get("error_type"),
                }
            else:
                # ✅ Normal success
                self.verified_cases[idx] = res
                self.tracking[str(idx + 1)] = {
                    "status": "success",
                    "verify": "Correct",
                    "message": "Case verified as correct.",
                }


                # If this index was previously in failed_list, remove it
                self.failed_list[:] = [f for f in self.failed_list if f["original_index"] != idx + 1]



    async def put(self, indices: List[int], results: List[Any]):
        """Workers call this to enqueue their results."""
        await self.queue.put((indices, results))

    async def stop(self):
        """Ask saver to stop and flush once more."""
        self._stop.set()
        await self.queue.put((None, None))

    def _flush_if_needed(self, force: bool = False):
        """Flush to disk periodically or on demand."""
        now = time.time()
        if force or (now - self._last_flush) >= SAVE_INTERVAL_SECONDS:
            save_json_atomic(self.verified_cases, VERIFIED_FILE)
            save_json_atomic(self.tracking, TRACKING_FILE)
            save_json_atomic(self.failed_list, FAILED_FILE)
            self._last_flush = now
         # Recalculate counts after applying repairs so repaired ones are marked as success immediately
            total = len(self.verified_cases)
            filled = sum(1 for v in self.verified_cases if v is not None)
            failed = sum(1 for v in self.verified_cases if is_fail_object(v))
            success = filled - failed
            print(f"💾 Saved: {filled}/{total} processed | ✅ success: {success} | ❌ failed: {failed} | 🧾 tracking entries: {len(self.tracking)-1 if '_verify_version' in self.tracking else len(self.tracking)}")


    async def run(self):
        """Main loop: apply results as they arrive and flush periodically."""
        print(f"🧷 Saver started (flush every {int(SAVE_INTERVAL_SECONDS)}s).")
        while True:
            indices, results = await self.queue.get()
            if indices is None and results is None:
                # Sentinel → final flush & exit
                self._flush_if_needed(force=True)
                print("🧷 Saver stopping (final flush complete).")
                return
            self._apply_results(indices, results)
            self._flush_if_needed(force=False)

# ====================== ORCHESTRATION ======================

def init_outputs(n: int) -> Tuple[List[Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Load or initialize verified_cases, tracking, failed_list."""
    verified_cases: List[Any] = load_json_safe(VERIFIED_FILE, [None] * n)
    if not isinstance(verified_cases, list) or len(verified_cases) != n:
        verified_cases = [None] * n

    tracking: Dict[str, Any] = load_json_safe(TRACKING_FILE, {})
    tracking["_verify_version"] = VERIFY_VERSION

    failed_list: List[Dict[str, Any]] = load_json_safe(FAILED_FILE, [])
    if not isinstance(failed_list, list):
        failed_list = []

    return verified_cases, tracking, failed_list

def indices_to_process(n: int, verified_cases: List[Any], tracking: Dict[str, Any]) -> List[int]:
    """
    Decide which indices still need processing (resume-friendly).
    A slot is considered DONE if:
      - tracking[k]["status"] == "success"
      - OR verified_cases[idx] is filled with a non-fail object
    Otherwise, it’s queued for processing.
    """
    todo: List[int] = []
    for i in range(n):
        k = str(i + 1)
        tracked = tracking.get(k, {})
        already_ok = tracked.get("status") == "success"
        slot = verified_cases[i]
        slot_ok = slot is not None and not is_fail_object(slot)
        if already_ok or slot_ok:
            continue
        todo.append(i)
    return todo

def build_batches_for_indices(indices: List[int], cases: List[Dict[str, Any]]) -> List[List[Tuple[int, Dict[str, Any]]]]:
    """Attach indices to cases and pack them into batches."""
    keyed = [(i, cases[i]) for i in indices]
    return dynamic_pack(keyed, BATCH_SIZE, MAX_CHARS_PER_REQUEST)

def needs_single_case_retry(obj: Any) -> bool:
    """
    Identify fails that deserve a targeted single-case retry:
      - invalid_json (model output formatting)
      - exception with retryable error text (429/quota/500/timeout/etc.)
    """
    if not is_fail_object(obj):
        return False
    et = (obj.get("error_type") or "").lower()
    if et == "invalid_json":
        return True
    if et == "exception" and is_retryable_exception_text(obj.get("reason", "")):
        return True
    return False

# ====================== BATCH RUNNER (one pass) ======================

async def run_pass(
    keyed_batches: List[List[Tuple[int, Dict[str, Any]]]],
    saver: Saver,
    pass_name: str,
    cycle_no: int,
) -> None:
    """
    Execute one pass over the provided batches.
    - Bounded by MAX_CONCURRENCY
    - For each batch result, apply targeted single-case retries on specific fails
    - Enqueue final results to Saver (single writer)
    """
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def single_case_retry(idx: int, case_obj: Dict[str, Any]) -> Any:
        """
        Try a single-case retry (tighter payload) with its own small retry loop.
        Used for invalid_json or retryable exception fails.
        """
        # Attempt up to MAX_RETRIES with backoff and honoring any server retry_delay
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Make the same verify call but with a one-item array
                results = await call_gemini_verify([case_obj])
                if results and len(results) == 1:
                    return results[0]
                return structured_fail("Single-case retry: unexpected result length", "single_retry_mismatch")
            except Exception as e:
                err = str(e)
                if attempt < MAX_RETRIES and is_retryable_exception_text(err):
                    wait_s = parse_retry_delay_seconds(err, default_wait=int(BASE_BACKOFF * attempt) + 2)
                    time.sleep(wait_s)
                    continue
                return structured_fail(f"Single-case retry exception: {err}", "single_retry_exception")

    async def worker(batch_pairs: List[Tuple[int, Dict[str, Any]]]):
        async with sem:
            indices = [i for i, _ in batch_pairs]
            cases = [c for _, c in batch_pairs]

            # 1) Batch call
            results = await call_gemini_verify(cases)

            # 2) Targeted single-case fallback for invalid_json / quota / 500
            #    (only for those still failing)
            repaired = False
            for j, res in enumerate(results):
                if needs_single_case_retry(res):
                    idx = indices[j]
                    case_obj = cases[j]
                    fixed = await single_case_retry(idx, case_obj)
                    # If fixed result is not a fail, or a different fail than before, use it
                    if fixed is not None:
                        results[j] = fixed
                        repaired = True

            # 3) Hand final results to the saver
            await saver.put(indices, results)

            # 4) Keep your console style: show batch range, size, and the cycle number
            print(f"✅ Batch done ({pass_name} | cycle {cycle_no}) → idx {indices[0]+1}..{indices[-1]+1} | size={len(indices)}{' | 🔧 repaired' if repaired else ''}")

    # Schedule all batches with limited concurrency
    print(f"🚀 {pass_name}: {len(keyed_batches)} batches scheduled with concurrency {MAX_CONCURRENCY} (cycle {cycle_no})")
    tasks = [asyncio.create_task(worker(batch)) for batch in keyed_batches]
    await asyncio.gather(*tasks)

# ====================== MAIN ======================

async def main_async() -> None:
    """Main entry (async). Runs multiple cycles until success or MAX_PASSES."""
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    ensure_genai_configured()

    # Load source cases
    cases: List[Dict[str, Any]] = load_json_safe(INPUT_FILE, [])
    if not isinstance(cases, list):
        raise ValueError("Input file is not a JSON array.")

    n = len(cases)
    print(f"🔍 Loaded {n} cases from {INPUT_FILE}")

    # Load or init outputs
    verified_cases, tracking, failed_list = init_outputs(n)
    saver = Saver(verified_cases, tracking, failed_list, cases)

    # Determine which indices need work (resume-friendly)
    todo = indices_to_process(n, verified_cases, tracking)
    if not todo:
        print("✅ Nothing to do. All cases already verified.")
        # Still flush version tag & files once
        save_json_atomic(verified_cases, VERIFIED_FILE)
        save_json_atomic(tracking, TRACKING_FILE)
        save_json_atomic(failed_list, FAILED_FILE)
        return

    print(f"📌 To process this run: {len(todo)} / {n} cases")

    # Start saver (single writer)
    saver_task = asyncio.create_task(saver.run())

    # ----------------------
    # CYCLE LOOP (1..MAX_PASSES)
    # ----------------------
    remaining = set(todo)  # indices still to try in the first cycle
    for cycle in range(1, MAX_PASSES + 1):
        # Build batches only from remaining indices
        keyed_batches = build_batches_for_indices(sorted(list(remaining)), cases)

        # First cycle uses the label "First pass", later cycles are "Retry pass"
        pass_name = "First pass" if cycle == 1 else "Retry pass"

        # Run this cycle over remaining
        await run_pass(keyed_batches, saver, pass_name=pass_name, cycle_no=cycle)

        # After a cycle, compute who remains failed/unfilled
        new_remaining: List[int] = []
        for i in remaining:
            v = verified_cases[i]
            if v is None or is_fail_object(v):
                new_remaining.append(i)

        # If nothing changed (no progress) OR nothing left → decide to continue/stop
        progress_made = (len(new_remaining) < len(remaining))
        remaining = set(new_remaining)

        if not remaining:
            # All done
            break

        if not progress_made and cycle < MAX_PASSES:
            # If no progress last cycle, wait a bit longer before next attempt.
            # This helps if rate limits were the reason.
            time.sleep(5)

    # Any still-None slots after MAX_PASSES → mark as exhausted
    fill_count = 0
    for i, v in enumerate(verified_cases):
        if v is None:
            verified_cases[i] = structured_fail("No result after retries", "exhausted_retries")
            saver.tracking[str(i + 1)] = {
                "status": "fail",
                "verify": "Incorrect",
                "reason": "No result after retries",
                "error_type": "exhausted_retries",
            }
            saver.failed_list.append({
                "original_index": i + 1,
                "reason": "No result after retries",
                "error_type": "exhausted_retries",
                "case": cases[i],
            })
            fill_count += 1
    if fill_count:
        print(f"⚠️ Filled {fill_count} unresolved slots with fail markers.")

    # Stop saver & final flush
    await saver.stop()
    await saver_task

    # Summary
    success = sum(1 for v in verified_cases if v is not None and not is_fail_object(v))
    failure = n - success
    print(f"\n🎯 Verification complete.")
    print(f"   ✅ Success: {success}")
    print(f"   ❌ Failed:  {failure}")
    print(f"   💾 Verified file: {VERIFIED_FILE}")
    print(f"   🧾 Tracking file: {TRACKING_FILE}")
    print(f"   🚩 Failed-only file: {FAILED_FILE}")

def main() -> None:
    """Sync wrapper to run the async main."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
