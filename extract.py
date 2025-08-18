import os
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import pdfplumber
import aiohttp
import math  # add at top of file if not already imported
import itertools


# ====================== CONFIG ======================
# Keep your original style/filenames; adjust BASE_DIR/PDF_BASE_PATH as needed.
BASE_DIR = r"C:\Users\Kushal\Desktop\new\data\court_36_29_jan"
PDF_BASE_PATH = BASE_DIR

CIVIL_OUTPUT   = r"C:\Users\Kushal\Desktop\new\civil_cases.json"
CRIMINAL_OUTPUT= r"C:\Users\Kushal\Desktop\new\criminal_cases.json"
PROGRESS_FILE  = r"C:\Users\Kushal\Desktop\new\processed_cases.json"

# Concurrency & batching
MAX_CONCURRENCY        = 5          # batches in flight
BATCH_MAX_ITEMS        = 3          # max cases per request
BATCH_MAX_CHARS        = 18000      # total chars per request (prompt + data)
PER_ITEM_TEXT_CAP      = 5000       # max chars from PDF per case
PDF_TIMEOUT            = 25

# Retries
RETRY_CYCLES           = 2
RETRY_BACKOFF_BASE     = 1.6
RETRY_BACKOFF_MAX      = 8.0

# Gemini 2.5 Flash Lite endpoint
GEMINI_API_KEY ="AIzaSyBFEIHo4GKXri8a3rNAaQKq0EwRSrwjwYs"
GEMINI_MODEL   = "gemini-2.5-flash-lite"

GEMINI_BASE    = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_URL = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"


SAVE_FLUSH_EVERY = 10

print("[CONFIG] BASE_DIR =", BASE_DIR)
print("[CONFIG] PDF_BASE_PATH =", PDF_BASE_PATH)
print("[CONFIG] CIVIL_OUTPUT =", CIVIL_OUTPUT)
print("[CONFIG] CRIMINAL_OUTPUT =", CRIMINAL_OUTPUT)
print("[CONFIG] PROGRESS_FILE =", PROGRESS_FILE)
print("[CONFIG] MAX_CONCURRENCY, BATCH_MAX_ITEMS, BATCH_MAX_CHARS =", MAX_CONCURRENCY, BATCH_MAX_ITEMS, BATCH_MAX_CHARS)
print("[CONFIG] PER_ITEM_PREVIEW_CAP =", PER_ITEM_TEXT_CAP)
print("[CONFIG] GEMINI_URL =", GEMINI_URL)



# ====================== HELPERS ======================
def _atomic_write_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _safe_load_json(path: str, default: Any):
    print(f"[SAFE_LOAD] Attempting to read: {path}")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default
            return json.loads(content)
    except Exception:
        print(f"[SAFE_LOAD] Failed to parse JSON; moved to {path}.bad")
        # Move aside corrupted tracking/output to avoid blocking the run
        try:
            os.replace(path, path + ".bad")
        except Exception:
            pass
        return default

def load_progress() -> Dict[str, Any]:
    d = _safe_load_json(PROGRESS_FILE, {})
    print(f"[PROGRESS] Loaded progress entries: {len(d)}")
    # optionally print first 5 entries
    if isinstance(d, dict) and d:
        sample = list(d.items())[:5]
        print("[PROGRESS] sample entries:", sample)
    return d if isinstance(d, dict) else {}

    # return _safe_load_json(PROGRESS_FILE, {})

def save_progress(d: Dict[str, Any]):
    print(f"[PROGRESS] Saving progress ({len(d)} entries) -> {PROGRESS_FILE}")
    _atomic_write_json(PROGRESS_FILE, d)

def discover_input_jsons():
    """
    Streaming discovery generator:
    - yields JSON file paths one by one
    - skips known output/progress filenames
    - prints progress every 1000 discovered files
    NOTE: this *yields* strings, not returns a list.
    """
    print(f"[DISCOVER] Streaming scan of BASE_DIR: {BASE_DIR}")
    count = 0
    skipped_outputs = {
        os.path.basename(CIVIL_OUTPUT),
        os.path.basename(CRIMINAL_OUTPUT),
        os.path.basename(PROGRESS_FILE),
    }

    for root, _, fs in os.walk(BASE_DIR):
        for f in fs:
            if not f.lower().endswith(".json"):
                continue
            base = os.path.basename(f)
            # skip the output files if present in the same folder
            if base in skipped_outputs:
                # print only occasionally to avoid too much noise
                if count % 1000 == 0:
                    print(f"[DISCOVER] Skipping output/progress file encountered: {os.path.join(root,f)}")
                continue
            count += 1
            if count % 1000 == 0:
                print(f"[DISCOVER] Discovered {count} JSON files so far (in {root})")
            yield os.path.join(root, f)

    print(f"[DISCOVER] Finished scanning BASE_DIR. Total discovered (iterated): {count}")


def clean_text(s: str) -> str:
    # Remove problematic control chars and overlong runs of whitespace
    s = s.encode("utf-8", "ignore").decode()
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ====================== PDF ======================
def _extract_pdf_text_worker(pdf_abs: str) -> str:
    if not pdf_abs or not os.path.exists(pdf_abs):
        return ""
    try:
        out = []
        with pdfplumber.open(pdf_abs) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text() or ""
                    if t:
                        out.append(t)
                except Exception:
                    pass
        return "\n\n".join(out)
    except Exception:
        return ""

async def extract_pdf_text(meta: dict, cap_chars: int) -> str:
    # meta may contain "pdf_link" or "pdf_path"
    link = meta.get("pdf_link") or meta.get("pdf_path") or ""
    print(f"[PDF] extract_preview_capped: case meta pdf_link={link!r}")
    pdf_abs = ""
    if link:
        # if only filename in link, resolve from base; if path, take basename and resolve
        pdf_abs = os.path.join(PDF_BASE_PATH, os.path.basename(link))
    if not pdf_abs or not os.path.exists(pdf_abs):
        print("[PDF] No pdf path found in metadata; returning 'Not available'")
        return "Not available"

    try:
        txt = await asyncio.wait_for(
            asyncio.to_thread(_extract_pdf_text_worker, pdf_abs),
            timeout=PDF_TIMEOUT
        )
        if not txt:
            return "Not available"
        txt = clean_text(txt)
        if len(txt) > cap_chars:
            original_len = len(txt)
            txt = txt[:cap_chars] + "\n[Truncated for length]"
            print(f"[PDF] Extracted {original_len} chars from {pdf_abs} (returning {len(txt)} chars after truncation).")


        return txt
    except asyncio.TimeoutError:
        print(f"[PDF] Extraction timeout for {pdf_abs} (timeout={PDF_TIMEOUT}s)")
        return "[PDF extraction timeout]"
    except Exception:
        print(f"[PDF] Error extracting text from {pdf_abs}: {repr(Exception)}")
        return "Not available"

# ====================== PROMPTS ======================
# IMPORTANT: Preserve your Civil/Criminal field structure exactly.
CIVIL_SCHEMA = {
    "case_type": "Civil",
    "contract_validity": "...",
    "property_rights": "...",
    "civil_procedure": "...",
    "court_direction": "...",
    "judgment_result": "..."
}
CRIMINAL_SCHEMA = {
    "case_type": "Criminal",
    "criminal_liability_analysis": "...",
    "evidence_evaluation": "...",
    "sentencing_considerations": "...",
    "procedural_compliance": "...",
    "judgment_result": "..."
}

def build_batch_prompt(batch_items: List[Dict[str, Any]]) -> str:
    header = f"""
You are an expert Indian legal judgment analyzer.

TASK:
For each case, read the provided metadata and extracted judgment text,
then produce a single JSON object with these top-level keys:
"case_id", "instruction", "question", "facts", "charge", "law", "law_content", "judgment", "category", "case_type".

### Rules for classification (VERY STRICT):

- **case_type must be either "Civil" or "Criminal".**

- Criminal = if the case involves:
  * FIRs, police, accused persons, arrests
  * IPC or CrPC sections (e.g., 498A, 420, 406, 302, 379, 506, etc.)
  * Special criminal laws: NDPS Act, POCSO, COTPA
  * Bail (regular/anticipatory), quash petitions, cognizance, criminal liability
  * Words like "accused", "offence", "charge sheet"

- Civil = if the case involves:
  * Property disputes, title deeds, injunctions, succession, tenancy
  * Contracts, specific performance, damages, family matters, company law
  * Land acquisition, tribunals (motor accident compensation, electricity disputes)
  * Constitutional writs under Article 226 (service matters, administrative disputes)
  * Words like "injunction", "property", "contract", "compensation", "writ petition"

- If both types of keywords appear:
  * Prefer **Criminal** if ANY clear criminal statute or concept (FIR, IPC, CrPC, NDPS, bail, accused, quash petition) is present.
  * Otherwise default to **Civil**.

### Other field rules:

- "case_id": copy exactly from input.
- "instruction": always "Parse and summarize this judgment".
- "question": a plain English question capturing the main legal issue.
- "facts": concise 2–4 line background of the case.
- "charge": the main legal issue or relief sought.
- "law": main Act(s)/Code(s) applied.
- "law_content": specific provision(s) explained in plain words (e.g., "Order XVIII Rule 17 CPC – Court may recall and examine a witness").
- "judgment": crisp summary of the final decision.
- "category": granular type (e.g., Civil: Property dispute, Civil: Contract dispute, Criminal: Bail, Criminal: Quash petition, etc.)

STRICT OUTPUT RULES:
- Output MUST be pure JSON: an array of JSON objects (one per case), no extra text.
- Every object MUST have a correct "case_type" field.

CASES_INPUT (array):
""".strip()

    return header + "\n" + json.dumps(batch_items, ensure_ascii=False, indent=2) + "\n\nReturn ONLY the JSON array."


# ====================== INSTRUCTION & QUESTION BUILDING ======================
DEFAULT_INSTRUCTION = "Summarize the facts, legal issue, provisions applied, and the High Court’s decision."

def build_question(rec: dict) -> str:
    charge = (rec.get("charge") or "").strip()
    lawc = (rec.get("law_content") or "").strip()

    if charge and lawc:
        return f"What did the High Court decide regarding {charge} under {lawc}?"
    if charge:
        return f"What did the High Court decide regarding {charge}?"
    if lawc:
        return f"What was the High Court’s ruling under {lawc}?"
    return "What was the High Court’s ruling in this case?"

def attach_instruction_and_question(records: list) -> list:
    out = []
    for r in records:
        enriched = dict(r)
        enriched["instruction"] = DEFAULT_INSTRUCTION
        enriched["question"] = build_question(enriched)
        out.append(enriched)
    return out


# ====================== GEMINI CALL ======================
async def call_gemini_json_array(session, prompt: str, batch_items: list):
    url = GEMINI_URL
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    try:
        async with session.post(url, json=payload, timeout=120) as resp:
            text = await resp.text()

            if resp.status != 200:
                return None, f"HTTP {resp.status} | {text[:200]}"

            try:
                data = await resp.json(content_type=None)
            except Exception as e:
                return None, f"invalid_json_response: {e} | raw={text[:200]}"

            try:
                cand = (data.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or [{}]
                raw_text = parts[0].get("text", "").strip()
            except Exception as e:
                return None, f"missing_fields: {e} | raw={text[:200]}"

            print(f"[DEBUG] Raw Gemini output snippet: {raw_text[:200]}")

            # --- Parse the model's raw JSON output ---
            try:
                parsed = json.loads(raw_text)
            except Exception:
                start = raw_text.find("{")
                if start == -1:
                    start = raw_text.find("[")
                if start != -1:
                    try:
                        parsed = json.loads(raw_text[start:])
                    except Exception:
                        parsed = {"raw_text": raw_text}
                else:
                    parsed = {"raw_text": raw_text}

            # Normalize into list of dicts
            if isinstance(parsed, dict):
                results = [parsed]
            elif isinstance(parsed, list):
                results = [item if isinstance(item, dict) else {"value": str(item)} for item in parsed]
            else:
                results = [{"value": str(parsed)}]

            # Ensure each result has a case_id
            for i, item in enumerate(results):
                if "case_id" not in item and i < len(batch_items):
                    item["case_id"] = batch_items[i].get("case_id")

            # ✅ FIX: only accept results that follow the new schema
            results = [r for r in results if ("facts" in r or "judgment" in r)]

            # Add instruction & question enrichment
            results = attach_instruction_and_question(results)

            print(f"[DEBUG] Normalized output type: {type(results)} | Length: {len(results)}")
            return results, None

    except Exception as e:
        return None, f"request_error: {e}"

# ====================== DATA CLASSES ======================
@dataclass
class CaseItem:
    case_id: str
    meta: Dict[str, Any]
    text: str

# ====================== BATCHING ======================
def pack_batches(items: List[CaseItem]) -> List[List[CaseItem]]:
    """
    Simple char-budget-aware packing:
    - up to BATCH_MAX_ITEMS per batch
    - total serialized chars <= BATCH_MAX_CHARS
    """
    batches: List[List[CaseItem]] = []
    current: List[CaseItem] = []
    current_len = 0

    def item_len(ci: CaseItem) -> int:
        # approximate serialized size inside batch json
        stub = {
            "case_id": ci.case_id,
            "instruction": ci.meta.get("instruction", "Parse and summarize this judgment"),
            "question": ci.meta.get("title", "Case Analysis"),
            "metadata": ci.meta,
            "text": ci.text,
        }
        return len(json.dumps(stub, ensure_ascii=False))

    for ci in items:
        ilen = item_len(ci)
        if not current:
            current = [ci]
            current_len = ilen
            continue
        if len(current) < BATCH_MAX_ITEMS and (current_len + ilen) <= BATCH_MAX_CHARS:
            current.append(ci)
            current_len += ilen
        else:
            batches.append(current)
            current = [ci]
            current_len = ilen
    if current:
        batches.append(current)
    return batches

# ====================== PIPELINE ======================
async def build_case_items(file_iter):
    """
    Async generator: given a synchronous iterable (generator) of file paths (file_iter),
    yields CaseItem for each file after loading metadata and extracting capped PDF text.
    This avoids building a full list in memory and starts producing items immediately.
    Usage:
        async for ci in build_case_items(discover_input_jsons()):
            ...
    """
    idx = 0
    for fp in file_iter:
        idx += 1
        case_id = os.path.basename(fp)
        # load metadata (safe)
        meta = _safe_load_json(fp, {})
        # extract a capped preview of pdf text
        txt = await extract_pdf_text(meta, PER_ITEM_TEXT_CAP)
        # yield the CaseItem
        yield CaseItem(case_id=case_id, meta=meta, text=txt)

    # finished
    print(f"[BUILD_ITEMS] Completed streaming case item generation (total yielded: {idx})")


def case_items_to_payloads(batch: List[CaseItem]) -> List[Dict[str, Any]]:
    payloads = []
    for ci in batch:
        instruction = ci.meta.get("instruction", "Parse and summarize this judgment")
        question    = ci.meta.get("title", "Case Analysis")
        payloads.append({
            "case_id": ci.case_id,
            "instruction": instruction,
            "question": question,
            "metadata": ci.meta,
            "text": ci.text
        })
    return payloads

async def process_batch(session: aiohttp.ClientSession, batch: List[CaseItem], queue: asyncio.Queue):
    # Build prompt for this batch
    payloads = case_items_to_payloads(batch)
    prompt = build_batch_prompt(payloads)

    # Retry loop
    delay = 1.0
    for attempt in range(RETRY_CYCLES + 1):
        # ⬇️ FIX 1: pass batch to call_gemini_json_array
        results, err = await call_gemini_json_array(session, prompt, batch)

        if results is not None:
            # Success: fan out results by case_id
            by_id = {r.get("case_id"): r for r in results if isinstance(r, dict)}
            for ci in batch:
                res = by_id.get(ci.case_id)

                # ⬇️ FIX 2: check for new schema instead of "answer"
                if res and ("facts" in res or "judgment" in res):
                    await queue.put({"kind": "success", "case_id": ci.case_id, "data": res})
                else:
                    await queue.put({"kind": "failure", "case_id": ci.case_id, "reason": "missing_case_result"})
            return

        # failure
        if attempt < RETRY_CYCLES and (err or "").lower() in {"timeout", "http 429", "network_error"}:
            await asyncio.sleep(min(delay, RETRY_BACKOFF_MAX))
            delay *= RETRY_BACKOFF_BASE
            continue

        # final failure -> mark each as failed with shared reason
        for ci in batch:
            await queue.put({"kind": "failure", "case_id": ci.case_id, "reason": err or "unknown_error"})
        return

async def result_writer(queue: asyncio.Queue):
    civil_cases   = _safe_load_json(CIVIL_OUTPUT, [])
    criminal_cases= _safe_load_json(CRIMINAL_OUTPUT, [])
    progress      = load_progress()
    since_flush   = 0

    while True:
        msg = await queue.get()
        if msg is None:
            _atomic_write_json(CIVIL_OUTPUT, civil_cases)
            _atomic_write_json(CRIMINAL_OUTPUT, criminal_cases)
            save_progress(progress)
            queue.task_done()
            break

        kind = msg.get("kind")
        case_id = msg.get("case_id")
        if kind == "success":
            res = msg["data"]

            # ✅ No more "answer" wrapper — use fields directly
            ctype = (res.get("case_type") or "").strip().lower()
            if ctype.startswith("crim"):
                criminal_cases.append(res)
            else:
                civil_cases.append(res)

            progress[case_id] = {
                "status": "success",
                "ts": datetime.now().isoformat(),
                "reason": "Processed"
            }
        else:
            progress[case_id] = {
                "status": "failed",
                "reason": msg.get("reason", "unknown"),
                "ts": datetime.now().isoformat()
            }

        since_flush += 1
        if since_flush >= SAVE_FLUSH_EVERY:
            _atomic_write_json(CIVIL_OUTPUT, civil_cases)
            _atomic_write_json(CRIMINAL_OUTPUT, criminal_cases)
            save_progress(progress)
            since_flush = 0

        queue.task_done()


# ====================== MAIN ======================

async def main_async():
    """
    Streaming main:
    - iterates discover_input_jsons() generator
    - skips already-successful cases using processed_cases.json (load_progress())
    - builds batches on the fly using the same char-budget logic
    - dispatches process_batch(session, batch, queue) tasks with a semaphore to limit concurrency
    - writer task runs concurrently and flushes results periodically
    """
    print("[MAIN] Starting streaming main_async()")
    file_iter = discover_input_jsons()  # generator now
    # load progress to skip already-successful cases
    progress = load_progress()  # returns dict
    processed_count_before = sum(1 for v in progress.values() if v.get("status") == "success")
    print(f"[MAIN] progress entries loaded={len(progress)}, already_success={processed_count_before}")

    queue = asyncio.Queue()
    writer_task = asyncio.create_task(result_writer(queue))

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = []                    # will hold created batch tasks
        current_batch: List[CaseItem] = []
        current_chars = 0
        total_discovered = 0
        total_queued = 0
        skipped = 0

        # helper local to estimate serialized size of an item (same logic as pack_batches)
        def estimate_item_len(ci: CaseItem) -> int:
            stub = {
                "case_id": ci.case_id,
                "instruction": ci.meta.get("instruction", "Parse and summarize this judgment"),
                "question": ci.meta.get("title", "Case Analysis"),
                "metadata": ci.meta,
                "text": ci.text,
            }
            return len(json.dumps(stub, ensure_ascii=False))

        # nested function to dispatch a batch with semaphore protection
        async def dispatch_batch(batch_to_send: List[CaseItem]):
            nonlocal total_queued
            if not batch_to_send:
                return
            total_queued += len(batch_to_send)
            print(f"[DISPATCH] Dispatching batch with {len(batch_to_send)} items. Total queued so far: {total_queued}")
            async def run(b=batch_to_send):
                async with sem:
                    await process_batch(session, b, queue)
            # schedule the batch worker immediately
            tasks.append(asyncio.create_task(run()))

        # STREAM files and form batches incrementally
        for fp in file_iter:
            total_discovered += 1
            case_id = os.path.basename(fp)
            # check progress: skip already-successful
            p = progress.get(case_id)
            if p and p.get("status") == "success":
                skipped += 1
                if skipped % 100 == 0:
                    print(f"[MAIN] Skipped {skipped} already-successful cases so far.")
                continue

            # load metadata and extract pdf text for this file
            meta = _safe_load_json(fp, {})
            text = await extract_pdf_text(meta, PER_ITEM_TEXT_CAP)
            ci = CaseItem(case_id=case_id, meta=meta, text=text)

            # estimate length for budget
            ilen = estimate_item_len(ci)

            # if can append to current batch, do so; else dispatch current batch and start a new one
            can_append = (len(current_batch) < BATCH_MAX_ITEMS) and ((current_chars + ilen) <= BATCH_MAX_CHARS)
            if can_append:
                current_batch.append(ci)
                current_chars += ilen
            else:
                # dispatch existing batch
                await dispatch_batch(current_batch)
                # start new batch with this item
                current_batch = [ci]
                current_chars = ilen

            # periodic small-status print
            if total_discovered % 500 == 0:
                print(f"[MAIN] Discovered {total_discovered} files (queued {total_queued}, skipped {skipped})")

        # after iteration, dispatch leftover batch
        if current_batch:
            await dispatch_batch(current_batch)

        # wait for all batch tasks to complete
        if tasks:
            print(f"[MAIN] Waiting for {len(tasks)} batch tasks to finish...")
            await asyncio.gather(*tasks)
        else:
            print("[MAIN] No batches were dispatched (no unprocessed cases found).")

    # tell writer to flush and exit
    await queue.put(None)
    await writer_task

    # final summary
    progress_after = load_progress()
    total_success_after = sum(1 for v in progress_after.values() if v.get("status") == "success")
    print(f"[MAIN] Streaming processing finished. Discovered={total_discovered}, skipped={skipped}, queued={total_queued}")
    print(f"[MAIN] Success before run={processed_count_before}, success after run={total_success_after}")
    print("[MAIN] All done.")


def main():
    if not GEMINI_API_KEY:
        raise RuntimeError("Set your GEMINI_API_KEY first (export as env var GEMINI_API_KEY)")
    print("[MAIN] Using Gemini endpoint:", GEMINI_URL)  # helpful debug
    asyncio.run(main_async())


if __name__ == "__main__":
    import traceback, sys, time
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print("[FATAL] Unhandled exception in main():", repr(e))
        traceback.print_exc()
        sys.exit(1)
    finally:
        print(f"[INFO] Script finished (elapsed {time.time() - start_time:.1f}s)")
