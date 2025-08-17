import json
import re
import time
import google.generativeai as genai
import os
from collections import Counter

# ---------- CONFIG ----------
INPUT_FILE = r"civil_cases.json"
OUTPUT_FILE = "gen_contract_disputes.json"
TRACKING_FILE = "gen_tracking.json"
GEMINI_MODEL = "gemini-2.5-flash-lite"
API_KEY = "AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM"

BATCH_SIZE = 5
MAX_INPUT_CHARS = 14000
WAIT_BETWEEN_REQUESTS = 5
MAX_RETRIES = 5
SAVE_INTERVAL = 5  # save output every N processed cases for speed

# ---------- SETUP ----------
genai.configure(api_key=API_KEY)

# ---------- PROMPT ----------
REWRITER_PROMPT = """
You are given an array of JSON case objects.
Process every case as a contract dispute.
For each case:
   - "instruction": Use exactly this text:
     "When giving your reply, first provide your reasoning process before the <DTK> tag, and then answer.
      Based on the following facts, charge, and legal provisions, summarize the unified legal reasoning and list all cited legal references."
   - "question": Write as:
     Facts: ...
     Charge: ...
     Law: ...
     Law content: ...
     (Use only info from original case)
   - "answer": An object with:
       "reasoning": Single paragraph combining facts, arguments, procedures, and judgment outcome.
       "reference": Dictionary with legal sections or case laws exactly as stated in judgment.
Only output a valid JSON array of the processed cases.
Do not include any explanations or text outside the JSON.
"""

# ---------- HELPERS ----------
def clean_json_response(raw_text):
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```json\s*", "", raw_text)
    raw_text = re.sub(r"```$", "", raw_text)
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    return match.group(0) if match else raw_text

def clean_newlines(obj):
    if isinstance(obj, str):
        return " ".join(obj.replace("\n", " ").split())
    elif isinstance(obj, list):
        return [clean_newlines(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: clean_newlines(v) for k, v in obj.items()}
    return obj

def load_tracking():
    try:
        if not os.path.exists(TRACKING_FILE):
            return {}
        with open(TRACKING_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_tracking(tracking):
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(tracking, f, ensure_ascii=False, indent=4)

def save_output(data, output_file):
    data = clean_newlines(data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_entries(entries, char_limit):
    entries_str = json.dumps(entries, ensure_ascii=False)
    if len(entries_str) > char_limit:
        entries_str = entries_str[:char_limit]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = genai.GenerativeModel(GEMINI_MODEL).generate_content(
                f"{REWRITER_PROMPT}\n\nHere is the JSON:\n{entries_str}"
            )
            raw_text = clean_json_response(response.text)
            processed_cases = json.loads(raw_text)

            # --- Change #2: normalize both single & batch outputs ---
            if isinstance(processed_cases, dict):
                processed_cases = [processed_cases]
            elif isinstance(processed_cases, list):
                processed_cases = [pc for pc in processed_cases if isinstance(pc, dict)]
            else:
                processed_cases = []

            # Enforce required structure
            required_keys = {"instruction", "question", "answer"}
            processed_cases = [
                {k: v for k, v in case.items() if k in required_keys}
                for case in processed_cases
            ]

            if not processed_cases:
                return False, "Empty or invalid response", []

            # Ensure 'reference' exists
            for case in processed_cases:
                if isinstance(case, dict) and "answer" in case and isinstance(case["answer"], dict):
                    if "reference" not in case["answer"] or not isinstance(case["answer"]["reference"], dict):
                        case["answer"]["reference"] = {}

            return True, None, clean_newlines(processed_cases)

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                time.sleep(WAIT_BETWEEN_REQUESTS * attempt)
            else:
                return False, "JSON parsing error", []
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                wait_time = (WAIT_BETWEEN_REQUESTS * attempt) + 2
                print(f"⏳ Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                if attempt < MAX_RETRIES:
                    time.sleep(WAIT_BETWEEN_REQUESTS * attempt)
                else:
                    return False, err_str, []

    return False, "Unknown error - no output from model", []

# ---------- MAIN ----------
def rewrite_dataset(input_file, output_file, batch_size=5, char_limit=14000):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracking = load_tracking()

    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    updated_data = json.loads(content)
                    if len(updated_data) < len(data):
                        updated_data.extend([{} for _ in range(len(data) - len(updated_data))])
                else:
                    updated_data = [{} for _ in range(len(data))]
        except json.JSONDecodeError:
            print("⚠ Output file corrupted. Starting fresh.")
            updated_data = [{} for _ in range(len(data))]
    else:
        updated_data = [{} for _ in range(len(data))]

    total_cases = len(data)
    processed_count = 0
    fail_reasons = []

    # 🔹 Only check {}
    pending_cases = [(str(i), entry) for i, entry in enumerate(data, 1) if updated_data[i-1] == {}]

    for i in range(0, len(pending_cases), batch_size):
        batch = pending_cases[i:i+batch_size]
        batch_keys = [k for k, _ in batch]
        batch_entries = [e for _, e in batch]
        print(f"🔄 Processing cases {batch_keys} ({i+1} to {i+len(batch)}) of {total_cases}...")
        success, fail_reason, processed_cases = process_entries(batch_entries, char_limit)

        for k, processed_case in zip(batch_keys, processed_cases if (success and processed_cases) else [{}] * len(batch_keys)):
            idx = int(k) - 1
            if success and processed_cases:
                updated_data[idx] = processed_case
                tracking[k] = {"status": "success"}
                print(f"✅ Case {k} processed successfully.")
            else:
                updated_data[idx] = {}
                tracking[k] = {"status": "fail", "reason": fail_reason or "Empty or invalid response"}
                fail_reasons.append(fail_reason or "Empty or invalid response")
                print(f"❌ Case {k} failed. Reason: {fail_reason or 'Empty or invalid response'}")

            processed_count += 1
            save_tracking(tracking)
            if processed_count % SAVE_INTERVAL == 0:
                save_output(updated_data, output_file)

    save_output(updated_data, output_file)
    success_count = sum(1 for case in updated_data if case != {})
    fail_count = total_cases - success_count
    print(f"✅ Processing complete. Output saved to {output_file}")
    print(f"Success: {success_count}, Failed: {fail_count}")
    if fail_reasons:
        reason_counts = Counter(fail_reasons)
        print("Failure reasons:", dict(reason_counts))

# ---------- RUN ----------
if __name__ == "__main__":
    rewrite_dataset(INPUT_FILE, OUTPUT_FILE)
