import json
import re
import os
import time
import random
import asyncio
import aiohttp
from collections import Counter
from tqdm import tqdm

# ---------- CONFIG ----------
MODE = "criminal"   # "civil" or "criminal"

CIVIL_INPUT    = r"C:\Users\Kushal\Desktop\new\civil_cases.json"
CIVIL_OUTPUT   = r"C:\Users\Kushal\Desktop\new\gen_civil.json"
CIVIL_TRACKING = r"C:\Users\Kushal\Desktop\new\gen_tracking_civil.json"

CRIMINAL_INPUT    = r"C:\Users\Kushal\Desktop\new\criminal_cases.json"
CRIMINAL_OUTPUT   = r"C:\Users\Kushal\Desktop\new\gen_criminal.json"
CRIMINAL_TRACKING = r"C:\Users\Kushal\Desktop\new\gen_tracking_criminal.json"

API_KEY ="AIzaSyCAEEIT9lc5WxLAyq_N4SaKouAggntDxC4"
GEMINI_MODEL   = "gemini-2.5-flash-lite"

MIN_CONCURRENCY = 2
MAX_CONCURRENCY = 3
BATCH_MAX_ITEMS = 5
BATCH_MAX_CHARS = 14000
RETRY_CYCLES = 5
BACKOFF_BASE = 2
TARGET_FAST = 8       # seconds per batch → speed up if faster
TARGET_SLOW = 15      # seconds per batch → slow down if slower


# ---------- PROMPT ----------
REWRITER_PROMPT = """
You are given an array of JSON case objects.

For each case, rewrite it into the following structure:

{
  "case_id": "<same as input>",
  "instruction": "Summarize the case and also answer the legal question.",
  "question": "<short decision-focused question derived from the 'charge' and 'law'>",
  "analysis": {
    "case_summary": {
      "facts": "<use 'facts' field>",
      "legal_issue": "<use 'charge' field>",
      "law_applied": "<use 'law' and 'law_content' fields>",
      "judgment": "<use 'judgment' field>"
    },
    "court_reasoning": {
      "reasoning": "<a single coherent paragraph combining facts, arguments, procedures, and outcome>",
      "decision": "<short final decision in plain words>"
    }
  },
  "metadata": {
    "category": "<use 'category'>",
    "case_type": "<use 'case_type'>",
    "law": "<primary law or statute>"
  }
}

Rules:
- Do not invent facts or provisions. Only use the input fields.
- The "question" should be clear, concise, and decision-focused (remove redundant law descriptions).
- Always include both "case_summary" and "court_reasoning" inside "analysis".
- The output JSON array must have exactly the same number of items as the input array, in the same order.
- Return ONLY a valid JSON array of processed cases. No explanations outside JSON.
"""




# ---------- HELPERS ----------
def clean_json_response(raw_text):
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"```$", "", raw_text)
    # Prefer array, else object
    m = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if m:
        return m.group(0)
    return raw_text


def normalize_case(input_case, output_case):
    return {
        "case_id": input_case.get("case_id", ""),
        "instruction": output_case.get("instruction") or "Summarize the case and also answer the legal question.",
        "question": output_case.get("question") or f"What did the High Court decide in relation to {input_case.get('charge','')}?",
        "analysis": {
            "case_summary": {
                "facts": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("facts")
                    or input_case.get("facts", "")
                ),
                "legal_issue": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("legal_issue")
                    or input_case.get("charge", "")
                ),
                "law_applied": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("law_applied")
                    or f"{input_case.get('law','')} {input_case.get('law_content','')}"
                ),
                "judgment": (
                    output_case.get("analysis", {})
                               .get("case_summary", {})
                               .get("judgment")
                    or input_case.get("judgment", "")
                )
            },
            "court_reasoning": {
                "reasoning": (
                    output_case.get("analysis", {})
                               .get("court_reasoning", {})
                               .get("reasoning")
                    or ""
                ),
                "decision": (
                    output_case.get("analysis", {})
                               .get("court_reasoning", {})
                               .get("decision")
                    or ""
                )
            }
        },
        "metadata": {
            "category": input_case.get("category", ""),
            "case_type": input_case.get("case_type", ""),
            "law": input_case.get("law", "")
        }
    }



def load_json_file(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        return default
    return default

def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ---------- GEMINI CALL ----------
async def call_gemini(session, entries):
    MODEL_TRY = [GEMINI_MODEL, "gemini-1.5-flash", "gemini-1.5-pro"]
    # Trim oversized fields to lower truncation risk
    def _clip(s, n): 
        return s if not isinstance(s, str) or len(s) <= n else (s[:n] + " ...[truncated]")
    slim = []
    for c in entries:
        d = dict(c)
        for k in ("facts", "law_content", "judgment"):
            if k in d:
                d[k] = _clip(d[k], 4000)  # ~4k chars per long field
        slim.append(d)

    prompt_text = f"{REWRITER_PROMPT}\n\nHere is the JSON:\n{json.dumps(slim, ensure_ascii=False)}"
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    headers = {"Content-Type": "application/json"}

    for model_name in MODEL_TRY:
        for attempt in range(RETRY_CYCLES):
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
                async with session.post(url, headers=headers, json=payload) as resp:
                    body_text = await resp.text()
                    if resp.status != 200:
                        # show first 1k chars for debugging
                        print(f"⚠️ HTTP {resp.status} from {model_name}: {body_text[:1000]}")
                        # Backoff only for 429/5xx; otherwise move to next model or fail fast
                        if resp.status == 429 or 500 <= resp.status < 600:
                            wait = BACKOFF_BASE * (2 ** attempt) + random.random()
                            print(f"⏳ Backing off {wait:.1f}s...")
                            await asyncio.sleep(wait)
                            continue
                        else:
                            break  # try next model
                    # status 200 -> parse JSON
                    data = json.loads(body_text)
                    # extract all text parts robustly
                    candidates = data.get("candidates") or []
                    if not candidates:
                        print("⚠️ No candidates in response")
                        break
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts and isinstance(candidates[0].get("content"), list):
                        # some responses wrap content as a list
                        parts = candidates[0]["content"][0].get("parts", [])
                    raw_text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    if not raw_text:
                        print(f"⚠️ Empty text in candidates for {model_name}")
                        break
                    fixed = clean_json_response(raw_text)
                    # Debug previews
                    print("🔍 Raw preview:", raw_text[:300].replace("\n", " "))
                    print("🔍 Cleaned preview:", fixed[:300].replace("\n", " "))
                    # load as list (accept object too)
                    try:
                        out = json.loads(fixed)
                        if isinstance(out, dict):
                            out = [out]
                        return out
                    except Exception as e:
                        print("⚠️ json.loads failed:", e)
                        # Try to salvage largest JSON array/object
                        salvage = re.search(r"\[.*\]", fixed, re.DOTALL) or re.search(r"\{.*\}", fixed, re.DOTALL)
                        if salvage:
                            try:
                                out = json.loads(salvage.group(0))
                                if isinstance(out, dict):
                                    out = [out]
                                return out
                            except:
                                pass
                        # Retry on next attempt
                        wait = BACKOFF_BASE * (2 ** attempt) + random.random()
                        print(f"⏳ Retrying in {wait:.1f}s due to invalid JSON...")
                        await asyncio.sleep(wait)
                        continue
            except Exception as e:
                wait = BACKOFF_BASE * (2 ** attempt) + random.random()
                print(f"⚠️ Error: {e} → retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
        # try next model if this one failed
    return None


# ---------- MAIN PIPELINE ----------
import time
from tqdm import tqdm

# dynamic concurrency settings
MIN_CONCURRENCY = 2
MAX_CONCURRENCY = 6
TARGET_FAST = 8       # seconds per round → speed up if faster
TARGET_SLOW = 15      # seconds per round → slow down if slower

async def process_cases(input_file, output_file, tracking_file):
    data = load_json_file(input_file, [])
    results = load_json_file(output_file, [])
    tracking = load_json_file(tracking_file, {})

    # filter cases to process: new or failed ones
    to_process = []
    for case in data:
        cid = case.get("case_id")
        if tracking.get(cid, {}).get("status") != "success":
            to_process.append(case)

    total_cases = len(to_process)
    print(f"📌 {total_cases} cases to process out of {len(data)} total")

    total_start = time.perf_counter()

    concurrency = MIN_CONCURRENCY   # start safe
    idx = 0

    async with aiohttp.ClientSession() as session:
        with tqdm(total=total_cases, desc="Processing cases") as pbar:
            while idx < total_cases:
                # pick concurrency number of batches
                active_batches = []
                batch_case_count = 0

                for _ in range(concurrency):
                    if idx >= total_cases:
                        break
                    batch = to_process[idx:idx+BATCH_MAX_ITEMS]
                    idx += BATCH_MAX_ITEMS
                    batch_case_count += len(batch)

                    batch_text = json.dumps(batch, ensure_ascii=False)
                    if len(batch_text) > BATCH_MAX_CHARS:
                        for case in batch:
                            active_batches.append([case])
                    else:
                        active_batches.append(batch)

                # run these batches in parallel
                start = time.perf_counter()
                await asyncio.gather(*[
                    process_batch(session, batch, results, tracking, output_file, tracking_file)
                    for batch in active_batches
                ])
                end = time.perf_counter()
                elapsed = end - start

                # update progress bar by number of cases processed in this round
                pbar.update(batch_case_count)

                # log speed & concurrency
                print(f"⚡ Concurrency={concurrency}, Round time={elapsed:.2f}s")

                # adjust concurrency dynamically
                if elapsed < TARGET_FAST and concurrency < MAX_CONCURRENCY:
                    concurrency += 1
                    print(f"⬆️ Increasing concurrency → {concurrency}")
                elif elapsed > TARGET_SLOW and concurrency > MIN_CONCURRENCY:
                    concurrency -= 1
                    print(f"⬇️ Decreasing concurrency → {concurrency}")

    total_end = time.perf_counter()
    elapsed_total = total_end - total_start

    success_count = sum(1 for v in tracking.values() if v.get("status") == "success")
    fail_count = sum(1 for v in tracking.values() if v.get("status") == "fail")
    print(f"\n🎯 Completed. Success: {success_count}, Fail: {fail_count}")
    print(f"⏱️ Total pipeline time: {elapsed_total:.2f} seconds")

    if total_cases:
        avg_time = elapsed_total / ((total_cases + BATCH_MAX_ITEMS - 1) // BATCH_MAX_ITEMS)
        print(f"⚡ Average time per batch: {avg_time:.2f} seconds")



async def process_batch(session, batch, results, tracking, output_file, tracking_file):
    ids = [c.get("case_id") for c in batch]
    print(f"\n🔄 Processing batch {ids}...")

    start = time.perf_counter()
    resp = await call_gemini(session, batch)
    end = time.perf_counter()
    elapsed = end - start
    print(f"⏱️ Batch {ids} took {elapsed:.2f} seconds")

    if resp and isinstance(resp, list):
        for case, out in zip(batch, resp):
            cid = case.get("case_id")
            norm = normalize_case(case, out)
            results.append(norm)
            tracking[cid] = {"status": "success"}
            print(f"✅ Case {cid} success")
    else:
        for case in batch:
            cid = case.get("case_id")
            tracking[cid] = {"status": "fail", "reason": "Model failed or invalid JSON"}
            print(f"❌ Case {cid} failed")

    save_json_file(output_file, results)
    save_json_file(tracking_file, tracking)
# ---------- RUN ----------
if __name__ == "__main__":
    if MODE == "civil":
        asyncio.run(process_cases(CIVIL_INPUT, CIVIL_OUTPUT, CIVIL_TRACKING))
    else:
        asyncio.run(process_cases(CRIMINAL_INPUT, CRIMINAL_OUTPUT, CRIMINAL_TRACKING))
