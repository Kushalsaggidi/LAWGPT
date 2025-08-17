import os
import json
import time
import re
from bs4 import BeautifulSoup

import signal
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional deps your environment already had
try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

"""
FINAL, SAFE, IDP-FRIENDLY BATCHER
---------------------------------
Key fixes vs. your last run:
1) **Denominator bug fixed** → progress now shows `/ TOTAL_FILES`, not `/ len(progress)`.
2) **Hard de-duplication** → both on load (repairs existing files) and on every append using a stable `case_key` from the question string (e.g., `CRLP/122/2025`).
3) **Schema + sanity checks** → results failing checks are marked failed + retried; never pollute output lists.
4) **Idempotent writes** → atomic JSON writes; cannot corrupt files if killed mid-run.
5) **Retry loops** → failed cases are re-batched up to N times; repairs increment a clear counter.
6) **Never re-count total** → `TOTAL = len(input_files)` is constant across prints.

Run modes:
- `python final.py` → normal process + retries
- `python final.py --dedupe-only` → just clean and rewrite `civil_cases.json` & `criminal_cases.json` then exit
- Configure paths below or via env vars.
"""

# =============================
# CONFIG
# =============================
BASE_DIR = os.environ.get("CASES_DIR", r"C:\\Users\\Kushal\\Desktop\\new\\data\\court_36_29_jan")
PDF_BASE_PATH = BASE_DIR

CIVIL_OUTPUT = r"C:\Users\Kushal\Desktop\new\civil_cases.json"
CRIMINAL_OUTPUT = r"C:\Users\Kushal\Desktop\new\criminal_cases.json"
PROGRESS_FILE = r"C:\Users\Kushal\Desktop\new\processed_cases.json"
# Batching + retries
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 2))
MAX_RETRY_CYCLES = int(os.environ.get("MAX_RETRY_CYCLES", 5))
# Throttling and limits
THROTTLE_SECONDS = float(os.environ.get("THROTTLE_SECONDS", 0.5))  # delay between API calls

# Backoff
BASE_BACKOFF_SECONDS = float(os.environ.get("BASE_BACKOFF_SECONDS", 2))
MAX_BACKOFF_SECONDS = float(os.environ.get("MAX_BACKOFF_SECONDS", 20))
IN_WORKER_MAX_ATTEMPTS = int(os.environ.get("IN_WORKER_MAX_ATTEMPTS", 4))

# Model/API config (adjust to your stack)
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini")  # or "openai" etc.
# API_KEY = "AIzaSyCz41qZibYl1rca9Ye7_pkKaaBlQnms6ME"

# =============================
# Helpers: IO (safe JSON)
# =============================

API_KEYS = [
    "AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM",
    "AIzaSyCz41qZibYl1rca9Ye7_pkKaaBlQnms6ME",
    "AIzaSyA27TjinZixsBBl1ZkY7dRV9dK5Hf-ZzbA",
    "AIzaSyBFEIHo4GKXri8a3rNAaQKq0EwRSrwjwYs",
    "AIzaSyCAEEIT9lc5WxLAyq_N4SaKouAggntDxC4",
]
current_key_index = 0
API_KEY = API_KEYS[current_key_index]
consecutive_failures = 0
FAILURE_THRESHOLD = 10

DEV_MODE = os.environ.get("DEV_MODE", "0").strip() in {"1", "true", "True"}

# =============================
# Helpers for API key rotation
# =============================

def process_batch(batch_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """Call the model for a batch, validate outputs for each input file.
    Returns mapping filename -> {status, data|reason}
    """
    global consecutive_failures, current_key_index, API_KEY  # declare globals at very start

    # Build input payload
    payload = []
    for fname in batch_files:
        case_data = build_case_payload_from_file(os.path.join(BASE_DIR, fname))
        if "_load_error" in case_data:
            # mark as failed directly without model call
            print(f"⚠️ Skipping {fname}: {case_data['_load_error']}")
            continue
        payload.append(case_data)

    prompt = build_prompt(payload)

    for attempt in range(1, IN_WORKER_MAX_ATTEMPTS + 1):
        result, err, wait = call_model(prompt)
        # Throttle between calls to avoid hitting quota too fast
        THROTTLE_SECONDS = 0.5  # you can increase if still hitting limits
        if THROTTLE_SECONDS > 0:
            time.sleep(THROTTLE_SECONDS)


        if result is not None:
            # Expect list of case dicts, *aligned* to our inputs by question key
            out: Dict[str, Dict[str, Any]] = {}

            # index predictions by case_key to align
            pred_by_key: Dict[str, Dict[str, Any]] = {}
            for item in result:
                if not isinstance(item, dict):
                    continue
                q = item.get("question", "")
                pred_by_key[case_key_from_question(q)] = item

            for fname, inp in zip(batch_files, payload):
                key = case_key_from_question(inp.get("question", ""))
                pred = pred_by_key.get(key)

                if not pred:
                    out[fname] = {"status": "failed", "reason": "no matching item returned"}
                else:
                    ok, reason, norm = validate_and_normalize_result(pred)
                    if not ok:
                        out[fname] = {"status": "failed", "reason": reason}
                    else:
                        out[fname] = {"status": "success", "data": norm}

                # ---- Failure tracking + API key rotation ----
                if out[fname]["status"] == "failed":
                    # Immediate rotate if rate-limit error
                    if "429" in out[fname]["reason"]:
                        current_key_index = (current_key_index + 1) % len(API_KEYS)
                        API_KEY = API_KEYS[current_key_index]
                        consecutive_failures = 0
                        print(f"🔑 Rotated API key to index {current_key_index} immediately due to HTTP 429")
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= FAILURE_THRESHOLD:
                            current_key_index = (current_key_index + 1) % len(API_KEYS)
                            API_KEY = API_KEYS[current_key_index]
                            consecutive_failures = 0
                            print(f"🔑 Rotated API key to index {current_key_index} after {FAILURE_THRESHOLD} consecutive failures")
                else:
                    consecutive_failures = 0
                # --------------------------------------------

            return out

        # error path → backoff
        if err and "429" in err:
            # Rotate immediately on HTTP 429 in error path
            current_key_index = (current_key_index + 1) % len(API_KEYS)
            API_KEY = API_KEYS[current_key_index]
            consecutive_failures = 0
            print(f"🔑 Rotated API key to index {current_key_index} due to HTTP 429 rate limit")

        if wait and wait > 0:
            time.sleep(min(wait, MAX_BACKOFF_SECONDS))
        else:
            sleep_s = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            time.sleep(sleep_s)

    # Exhausted attempts
    return {fname: {"status": "failed", "reason": "model_error: " + (err or "unknown")} for fname in batch_files}


def rotate_api_key(reason: str = ""):
    """Rotate to the next API key and reset failure counter."""
    global current_key_index, API_KEY, consecutive_failures
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    API_KEY = API_KEYS[current_key_index]
    consecutive_failures = 0
    if reason:
        print(f"🔑 Rotated API key to index {current_key_index} ({reason})")
    else:
        print(f"🔑 Rotated API key to index {current_key_index}")


def _atomic_write_json(path: str, data: Any):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


def _safe_load_json(path: str, default: Any):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print(f"⚠️  Warning: {path} had invalid JSON. Keeping a backup and resetting to default.")
        try:
            os.replace(path, path + ".bad")
        except Exception:
            pass
        return default


def load_progress() -> Dict[str, Any]:
    return _safe_load_json(PROGRESS_FILE, {})


def save_progress(d: Dict[str, Any]):
    _atomic_write_json(PROGRESS_FILE, d)


# =============================
# Helpers: case key, validation, classification
# =============================

def case_key_from_question(q: str) -> str:
    """Stable key from the text inside quotes, before ' of '.
    Example question:
    "Parse the judgment text of 'CRLP/122/2025 of Sharon Construction, Vs The State of Telangana'."
    → key = 'CRLP/122/2025'
    Falls back to whole question if pattern not found.
    """
    if not q:
        return ""
    m = re.search(r"['\"]([^'\"]+)['\"]", q)
    if not m:
        return q.strip()
    inner = m.group(1)
    key = inner.split(" of ", 1)[0].strip()
    return key or q.strip()


def infer_case_type_from_question(q: str) -> str:
    q_upper = (q or "").upper()

    # Keywords strongly indicating criminal cases
    criminal_keywords = [
        "CRL", "CRLP", "C.C.", "CRIM", "CRIMINAL", "FIR", "IPC", "BNS", "BNSS",
        "P.S.", "POLICE STATION", "NDPS", "SC/ST ACT", "ARMS ACT",
        "ACCUSED", "CHARGE SHEET", "REMAND", "QUASH"
    ]

    if any(tag in q_upper for tag in criminal_keywords):
        return "Criminal"
    return "Civil"



def validate_and_normalize_result(obj: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Ensure minimum schema; coerce/normalize obvious fields.
    Returns (is_valid, reason_if_invalid, normalized_obj)
    """
    if not isinstance(obj, dict):
        return False, "result not a dict", obj

    # required top-level keys
    if not all(k in obj for k in ("instruction", "question", "answer")):
        return False, "missing top-level keys", obj

    q = obj.get("question")
    ans = obj.get("answer")
    if not isinstance(q, str) or not q.strip():
        return False, "empty question", obj
    if not isinstance(ans, dict):
        return False, "answer is not an object", obj

    # case_type inference and normalization
    case_type = ans.get("case_type")
    inferred = infer_case_type_from_question(q)

    if not isinstance(case_type, str) or case_type.strip() == "":
        ans["case_type"] = inferred
    else:
        if case_type not in ("Civil", "Criminal"):
            ans["case_type"] = inferred
        elif case_type != inferred:
            ans["case_type"] = inferred
            ans["_note_case_type_normalized"] = True

    # Normalize every field in answer to string (except the special note flag)
    for k, v in list(ans.items()):
        if k == "_note_case_type_normalized":
            continue
        if v is None:
            ans[k] = "Not applicable"
        elif not isinstance(v, str):
            try:
                ans[k] = json.dumps(v, ensure_ascii=False)  # stringify complex types
            except Exception:
                ans[k] = str(v)
        ans[k] = ans[k].strip()

    # Must have a non-empty judgment_result
    jr = ans.get("judgment_result")
    if not isinstance(jr, str) or not jr.strip():
        return False, "empty judgment_result", obj

    # Filter out LLM disclaimers
    flat = json.dumps(ans, ensure_ascii=False).lower()
    if "as an ai" in flat or "cannot access external" in flat:
        return False, "ai disclaimer in answer", obj

    return True, None, obj



# =============================
# Prompting + model call (replace with your stack if needed)
# =============================

def extract_question_from_metadata(raw):
    """Generate a clean question from metadata raw_html and court_name."""
    soup = BeautifulSoup(raw.get("raw_html", ""), "html.parser")
    btn = soup.find("button")
    if btn:
        text = btn.get_text(strip=True)
        case_title = re.sub(r'\s+', ' ', text)
        return f"Parse the judgment text of '{case_title}' from the {raw.get('court_name', '').strip()}."
    return ""

def build_prompt(batch_cases: list[dict]) -> str:
    """
    Build a strict prompt for Gemini to output exactly what we need.
    """
    formatted_cases = []
    for case in batch_cases:
        formatted_cases.append({
            "case_id": case.get("case_id", ""),
            "metadata": {k: v for k, v in case.items() if k not in ("case_id", "pdf_preview")},
            "pdf_preview": case.get("pdf_preview", "")
        })

    return f"""
You are a legal case summarizer.

For each input case, read the "metadata" and "pdf_preview".
Classify the case as either "Civil" or "Criminal".
Return ONLY a JSON array, with each object having this exact structure:

For Criminal cases:
{{
    "case_id": "<exact case_id from input>",
    "answer": {{
        "case_type": "Criminal",
        "criminal_liability_analysis": "...",
        "evidence_evaluation": "...",
        "sentencing_considerations": "...",
        "procedural_compliance": "...",
        "judgment_result": "..."
    }}
}}

For Civil cases:
{{
    "case_id": "<exact case_id from input>",
    "answer": {{
        "case_type": "Civil",
        "contract_validity": "...",
        "property_rights": "...",
        "civil_procedure": "...",
        "court_direction": "...",
        "judgment_result": "..."
    }}
}}

Rules:
- Output must be a single JSON array containing one object per input case.
- The "case_id" must match the input exactly, character-for-character, including the ".json" extension.
- Do not skip, omit, or reorder cases.
- If a section is not applicable, write "Not applicable".
- Use only the provided metadata and PDF preview for the summary.

Cases to process:
{json.dumps(formatted_cases, ensure_ascii=False)}
""".strip()




def call_model(prompt: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[float]]:
    global current_key_index, API_KEY, consecutive_failures  # Move to very top of function

    """Return (case_list, error_message, retry_after_seconds).
    Implemented for Gemini JSON output; adapt if you use another provider.
    """
    if MODEL_PROVIDER == "gemini" and requests is not None:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        params = {"key": API_KEY}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        try:
            resp = requests.post(url, params=params, json=payload, timeout=60)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    cand = (data.get("candidates") or [{}])[0]
                    parts = ((cand.get("content") or {}).get("parts") or [{}])
                    text = (parts[0] or {}).get("text", "")
                    start_idx = text.find("[")
                    end_idx = text.rfind("]")
                    raw_json = text[start_idx:end_idx+1] if start_idx != -1 and end_idx != -1 else text
                    raw_json = raw_json.replace(",]", "]").replace(", }", "}")
                    parsed = json.loads(raw_json)
                    if isinstance(parsed, dict):
                        parsed = [parsed]
                    if not isinstance(parsed, list):
                        return None, "model returned non-list JSON", None
                    return parsed, None, None
                except Exception as e:
                    return None, f"parse_error: {e}", None

            # non-200: try retry-after header

            elif resp.status_code in (429, 503):
                wait_time = int(resp.headers.get("Retry-After", 5))  # try to respect API's suggestion
                print(f"⚠️ HTTP {resp.status_code} from API. Waiting {wait_time} seconds...")

                # Only rotate key for 429 if we haven't tried all keys yet in this burst
                if resp.status_code == 429 and consecutive_failures >= FAILURE_THRESHOLD:
                    current_key_index = (current_key_index + 1) % len(API_KEYS)
                    API_KEY = API_KEYS[current_key_index]
                    consecutive_failures = 0
                    print(f"🔑 Rotated API key to index {current_key_index} due to repeated HTTP 429 rate limit")

                time.sleep(wait_time)
                return None, f"http_{resp.status_code}", wait_time


            ra = resp.headers.get("Retry-After")
            wait = None
            try:
                if ra:
                    wait = float(ra)
            except Exception:
                wait = None
            return None, f"http_{resp.status_code}", wait
        except requests.RequestException as e:
            return None, f"network_error: {e}", None
    else:
        # Stub for offline/dev testing – treat as failure to enable retries
        return None, "no_model_provider_configured", None


# =============================
# PDF helpers (optional, if you attach pdf_text per case)
# =============================

def extract_pdf_text(pdf_path: str) -> str:
    if not pdf_path or pdfplumber is None:
        return ""
    if not os.path.exists(pdf_path):
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            out = []
            for page in pdf.pages:
                try:
                    out.append(page.extract_text() or "")
                except Exception:
                    pass
        return "\n\n".join(s for s in out if s)
    except Exception:
        return ""


# =============================
# Batching worker
# =============================

def iter_batches(items: List[str], k: int) -> List[List[str]]:
    return [items[i : i + k] for i in range(0, len(items), k)]


def build_case_payload_from_file(json_path: str) -> Dict[str, Any]:
    """Read input .json metadata and build a case object for the model."""
    raw = _safe_load_json(json_path, {})
    case: Dict[str, Any] = {
        "instruction": raw.get("instruction") or "Extract the appropriate legal sections from this judgment.",
        "question": raw.get("question") or raw.get("title") or os.path.basename(json_path),
    }

    # Support pdf_path, pdf, and pdf_link
    pdf_rel = raw.get("pdf_path") or raw.get("pdf") or raw.get("pdf_link") or ""
    if pdf_rel:
        # pdf_link in your metadata is just a relative filename
        pdf_abs = os.path.join(PDF_BASE_PATH, os.path.basename(pdf_rel))

        if not os.path.exists(pdf_abs):
            print(f"⚠️ Missing PDF: {pdf_abs}")
            case["_load_error"] = "PDF missing"
            return case

        preview = extract_pdf_text(pdf_abs)
        if not preview:
            print(f"⚠️ PDF read error: {pdf_abs}")
            case["_load_error"] = "PDF read error"
            return case

        case["pdf_preview"] = preview[:25000]  # Limit preview length

    # Ensure every case has a case_id for matching
    if not case.get("case_id"):
        case["case_id"] = os.path.basename(json_path)

    # Always return case, even if no PDF found
    return case

   
# =============================
# Main runner
# =============================

def summarize_counts(progress: Dict[str, Any], total_files: int) -> Tuple[int, int, int]:
    succ = sum(1 for v in progress.values() if v.get("status") == "success")
    fail = sum(1 for v in progress.values() if v.get("status") == "failed")
    return total_files, succ, fail


def load_and_dedupe_outputs() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set, set, int, int]:
    civil = _safe_load_json(CIVIL_OUTPUT, [])
    crim = _safe_load_json(CRIMINAL_OUTPUT, [])

    seen_civil = set()
    seen_crim = set()

    def dedupe(lst: List[Dict[str, Any]], seen: set) -> List[Dict[str, Any]]:
        deduped = []
        for item in lst:
            q = (item or {}).get("question", "")
            key = case_key_from_question(q)
            if key and key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    civil_d = dedupe([x for x in civil if isinstance(x, dict)], seen_civil)
    crim_d = dedupe([x for x in crim if isinstance(x, dict)], seen_crim)

    if len(civil_d) != len(civil):
        print(f"🧹 Repaired duplicates in civil: {len(civil)} → {len(civil_d)}")
    if len(crim_d) != len(crim):
        print(f"🧹 Repaired duplicates in criminal: {len(crim)} → {len(crim_d)}")

    _atomic_write_json(CIVIL_OUTPUT, civil_d)
    _atomic_write_json(CRIMINAL_OUTPUT, crim_d)

    return civil_d, crim_d, seen_civil, seen_crim, len(civil) - len(civil_d), len(crim) - len(crim_d)


def run_pass(pass_name: str, files_subset: List[str], progress: Dict[str, Any],
             civil_cases: List[Dict[str, Any]], criminal_cases: List[Dict[str, Any]],
             seen_civil: set, seen_crim: set, TOTAL: int):

    if not files_subset:
        print(f"✅ {pass_name}: nothing to do.")
        return

    batches = iter_batches(files_subset, BATCH_SIZE)
    print(f"🚀 {pass_name}: {len(batches)} batches scheduled with concurrency {MAX_WORKERS}")

    for batch in batches:
        # fan-out to worker pool (single task per batch keeps memory flat)
        results = process_batch(batch)

        repairs_in_batch = 0
        successes_in_batch = 0

        for fname in batch:
            r = results.get(fname) or {"status": "failed", "reason": "no result"}
            prev_status = (progress.get(fname) or {}).get("status")

            if r["status"] == "success":
                ec = r["data"]
                q = ec.get("question", "")
                key = case_key_from_question(q)
                bucket = (ec.get("answer", {}) or {}).get("case_type", "Civil")

                if bucket == "Civil":
                    if key not in seen_civil:
                        civil_cases.append(ec)
                        seen_civil.add(key)
                else:
                    if key not in seen_crim:
                        criminal_cases.append(ec)
                        seen_crim.add(key)

                progress[fname] = {"status": "success", "failure_reason": None}
                successes_in_batch += 1
                if prev_status == "failed":
                    repairs_in_batch += 1
            else:
                progress[fname] = {"status": "failed", "failure_reason": r.get("reason", "Unknown")}

        # Save after each batch (atomic)
        _atomic_write_json(CIVIL_OUTPUT, civil_cases)
        _atomic_write_json(CRIMINAL_OUTPUT, criminal_cases)
        save_progress(progress)

        total, succ, fail = summarize_counts(progress, TOTAL)
        b_start, b_end = batch[0], batch[-1]
        repaired_flag = f" | 🔧 repaired {repairs_in_batch}" if repairs_in_batch else ""
        print(
            f"✅ Batch done ({pass_name}) → {b_start}..{b_end} | size={len(batch)}"
            f"{repaired_flag} | 📈 now: ✅ {succ} | ❌ {fail} / {total}"
        )

    # end-of-pass summary
    total, succ, fail = summarize_counts(progress, TOTAL)
    print(f"\n📊 {pass_name} summary → ✅ {succ} | ❌ {fail} / {total}")


# =============================
# CLI
# =============================

def main():
    # Graceful Ctrl+C for multiproc environments
    try:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    except Exception:
        pass

    # Discover input json files (exclude our outputs)
    all_files = sorted(
        [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".json")
         and os.path.basename(f) not in {
            os.path.basename(CIVIL_OUTPUT),
            os.path.basename(CRIMINAL_OUTPUT),
            os.path.basename(PROGRESS_FILE),
        }]
    )

    if not all_files:
        print("⚠️  No input JSON case files found. Exiting.")
        return

    TOTAL = len(all_files)
    print(f"🔍 Loaded {TOTAL} case files")

    # Load + dedupe outputs (also repairs any existing duplicates)
    civil_cases, criminal_cases, seen_civil, seen_crim, cfix, xfix = load_and_dedupe_outputs()

    # Quick exit mode to only dedupe
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--dedupe-only":
        print("✅ Dedupe-only finished.")
        return

    # Progress
    progress: Dict[str, Any] = load_progress()

    # First pass set: everything not marked success
    pending = [f for f in all_files if (progress.get(f) or {}).get("status") != "success"]
    print(f"📌 First pass target: {len(pending)} cases")

    # First pass
    run_pass("First pass", pending, progress, civil_cases, criminal_cases, seen_civil, seen_crim, TOTAL)

    # Retry failed up to N cycles
    for cycle in range(1, MAX_RETRY_CYCLES + 1):
        failed_files = [fname for fname, st in progress.items() if (st or {}).get("status") == "failed"]
        if not failed_files:
            print("🎉 All cases processed successfully after retries!")
            break
        print(f"\n🔄 Retry cycle {cycle} | remaining failed: {len(failed_files)}")
        run_pass(f"Retry pass {cycle}", failed_files, progress, civil_cases, criminal_cases, seen_civil, seen_crim, TOTAL)

    # Final counts
    total, succ, fail = summarize_counts(progress, TOTAL)
    print(f"\n🏁 Finished. Total: {total} | ✅ success: {succ} | ❌ failed: {fail}")


if __name__ == "__main__":
    main()
