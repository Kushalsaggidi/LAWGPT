import json
import re
import time
import os
import google.generativeai as genai

# ---- CONFIG ----
INPUT_FILE = "gen_contract_disputes.json"
OUTPUT_FILE = "cases_polished.json"
TRACKING_FILE = "polish_tracking.json"
GEMINI_MODEL = "gemini-2.5-flash-lite"
API_KEY = "AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM"

MAX_CHARS_PER_REQUEST = 14000
BATCH_SIZE = 5
MAX_RETRIES = 2
BASE_WAIT_SECONDS = 5

genai.configure(api_key=API_KEY)

POLISH_PROMPT = """
You are given an array of JSON legal case objects. For each case:

1. Verify correctness, consistency, and factual accuracy.
2. Ensure reasoning matches facts, charge, and law references.
3. Fix hallucinations, missing or wrong references, or inconsistencies.
4. Fill empty references with relevant legal provisions based on facts, charges, and law content.
5. Maintain exact structure:
   {
       "instruction": "...",
       "question": "...",
       "answer": {
           "reasoning": "...",
           "reference": {...}
       }
   }
6. No field should be empty.
7. Return ONLY the JSON array, no other text.
"""

# ---- Helpers ----

def load_json_safe(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error reading {path}, starting fresh.")
        return {}

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def clean_text(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def normalize_newlines(obj):
    if isinstance(obj, str):
        return " ".join(obj.split())
    elif isinstance(obj, list):
        return [normalize_newlines(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: normalize_newlines(v) for k, v in obj.items()}
    return obj

def to_required_format(data):
    if isinstance(data, dict):
        data = [data]
    formatted = []
    for case in data:
        if not isinstance(case, dict):
            continue
        formatted.append({
            "instruction": case.get("instruction", "").strip(),
            "question": case.get("question", "").strip(),
            "answer": {
                "reasoning": case.get("answer", {}).get("reasoning", "").strip(),
                "reference": case.get("answer", {}).get("reference", {})
            }
        })
    return formatted

# ---- Gemini API with 429 handling ----
def call_gemini_api(cases):
    payload = f"{POLISH_PROMPT}\n\nHere is the JSON:\n{json.dumps(cases, ensure_ascii=False)}"
    while True:
        try:
            response = genai.GenerativeModel(GEMINI_MODEL).generate_content(payload)
            cleaned = clean_text(response.text or "")
            if not cleaned.strip():
                raise ValueError("Empty response from Gemini")
            parsed = json.loads(cleaned)
            parsed = to_required_format(parsed)
            for case in parsed:
                if (not case["instruction"] or 
                    not case["question"] or 
                    not case["answer"]["reasoning"] or 
                    not isinstance(case["answer"]["reference"], dict) or 
                    not case["answer"]["reference"]):
                    raise ValueError("Invalid or empty field in Gemini output")
            return parsed

        except Exception as e:
            err_str = str(e)
            if "429" in err_str and "quota" in err_str.lower():
                match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_str)
                if match:
                    wait_time = int(match.group(1)) + 2
                else:
                    wait_time = 15
                print(f"⚠️ Rate limit hit. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            raise

def batch_cases_by_char_limit(cases, max_chars):
    batch = []
    total_len = 0
    for case in cases:
        case_str = json.dumps(case, ensure_ascii=False)
        if total_len + len(case_str) > max_chars and batch:
            yield batch
            batch = [case]
            total_len = len(case_str)
        else:
            batch.append(case)
            total_len += len(case_str)
    if batch:
        yield batch

def update_polished_data(polished_data, key, new_case):
    idx = int(key) - 1
    while len(polished_data) <= idx:
        polished_data.append({})
    polished_data[idx] = new_case

# ---- Main polishing function ----

def polish_dataset():
    print("Loading input cases...")
    cases = load_json_safe(INPUT_FILE)
    if not isinstance(cases, list):
        print("Input file is not a JSON array.")
        return

    tracking = load_json_safe(TRACKING_FILE)
    polished_data = load_json_safe(OUTPUT_FILE)
    if not isinstance(polished_data, list) or len(polished_data) != len(cases):
        polished_data = [{} for _ in range(len(cases))]

    candidates = []
    for i in range(len(cases)):
        key = str(i+1)
        if polished_data[i] == {} or (tracking.get(key, {}).get("status") == "fail"):
            candidates.append((key, cases[i]))

    if not candidates:
        print("No cases to process.")
        return

    print(f"Processing {len(candidates)} cases in dynamic batches...")

    for i in range(0, len(candidates), BATCH_SIZE):
        batch_keys, batch_cases = zip(*candidates[i:i+BATCH_SIZE])
        batch_cases = list(batch_cases)

        for sub_batch in batch_cases_by_char_limit(batch_cases, MAX_CHARS_PER_REQUEST):
            sub_batch_keys = [batch_keys[batch_cases.index(c)] for c in sub_batch]
            try:
                polished = call_gemini_api(sub_batch)
                polished = normalize_newlines(polished)
                for key, polished_case in zip(sub_batch_keys, polished):
                    update_polished_data(polished_data, key, polished_case)
                    tracking[key] = {"status": "success"}
                print(f"Batch {sub_batch_keys} polished successfully.")
            except Exception as e:
                print(f"Batch failed: {sub_batch_keys}, trying individual cases...")
                for key, case in zip(sub_batch_keys, sub_batch):
                    try:
                        polished_single = call_gemini_api([case])
                        polished_single = normalize_newlines(polished_single)
                        update_polished_data(polished_data, key, polished_single[0])
                        tracking[key] = {"status": "success"}
                        print(f"Case {key} polished successfully on individual retry.")
                    except Exception as ex:
                        tracking[key] = {"status": "fail", "reason": str(ex)}
                        print(f"Case {key} failed after individual retry: {ex}")

            save_json(tracking, TRACKING_FILE)
            save_json(polished_data, OUTPUT_FILE)

    print(f"✅ Polishing complete. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    polish_dataset()
