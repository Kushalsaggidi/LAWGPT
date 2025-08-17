import json
import re
import time
import google.generativeai as genai
import os

# ---------- CONFIG ----------
INPUT_FILE = "contract_disputes.json"
OUTPUT_FILE = "cases_polished.json"
TRACKING_FILE = "polish_tracking.json"
SOURCE_TRACKING_FILE = "tracking.json"
GEMINI_MODEL = "gemini-2.5-flash-lite"
API_KEY = "AIzaSyCz41qZibYl1rca9Ye7_pkKaaBlQnms6ME"

BATCH_SIZE = 5
MAX_INPUT_CHARS = 14000
WAIT_BETWEEN_REQUESTS = 5

# ---------- SETUP ----------
genai.configure(api_key=API_KEY)

POLISH_PROMPT = """
You are given an array of JSON legal case objects. For each case:
1. Verify the correctness, consistency, and factual accuracy of the data.
2. Check if the reasoning matches the facts, charge, and law references.
3. If you find any hallucination, missing or wrong references, or inconsistencies, fix them.
4. Return a polished JSON array of the same cases, with corrected and verified data.
5. Maintain the same structure:
   - "instruction": the original instruction text,
   - "question": original question text,
   - "answer": object with:
        "reasoning": a logically consistent summary paragraph,
        "reference": dictionary of legal references cited exactly as in the original.
6. Do not include any text outside the JSON array.
"""

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

def load_tracking(file_path):
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
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
    for attempt in range(2):
        try:
            response = genai.GenerativeModel(GEMINI_MODEL).generate_content(
                f"{POLISH_PROMPT}\n\nHere is the JSON:\n{entries_str}"
            )
            raw_text = clean_json_response(response.text)
            processed_cases = json.loads(raw_text)
            if isinstance(processed_cases, dict):
                processed_cases = [processed_cases]
            processed_cases = clean_newlines(processed_cases)
            return True, None, processed_cases
        except json.JSONDecodeError:
            if attempt == 0:
                time.sleep(WAIT_BETWEEN_REQUESTS)
            else:
                return False, "JSON parsing error", []
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                time.sleep(65)
            else:
                if attempt == 0:
                    time.sleep(WAIT_BETWEEN_REQUESTS)
                else:
                    return False, err_str, []

def polish_dataset(input_file, output_file, batch_size=5, char_limit=14000):
    # Load all data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracking = load_tracking(TRACKING_FILE)

    # Load existing polished output
    polished_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    polished_data = json.loads(content)
        except json.JSONDecodeError:
            polished_data = []

    # Manual loop from case 653 to 1016
    for start in range(653, 1017, batch_size):
        end = min(start + batch_size, 1017)
        batch_keys = list(range(start, end))

        # Get the entries from your data — careful: index = key - 1
        batch_entries = []
        for key in batch_keys:
            if key - 1 < len(data):
                batch_entries.append(data[key - 1])
            else:
                # If case doesn't exist in file, make empty placeholder
                batch_entries.append({})

        print(f"Processing cases: {batch_keys}")
        success, fail_reason, processed_cases = process_entries(batch_entries, char_limit)
        if success:
            for key, case in zip(batch_keys, processed_cases):
                idx = key - 1
                if idx < len(polished_data):
                    polished_data[idx] = case
                else:
                    while len(polished_data) < idx:
                        polished_data.append({})
                    polished_data.append(case)
                tracking[str(key)] = {"status": "success"}
        else:
            for key in batch_keys:
                tracking[str(key)] = {"status": "fail", "reason": fail_reason}

        save_tracking(tracking)
        save_output(polished_data, output_file)

    print(f"✅ Polishing complete. Output saved to {output_file}")


if __name__ == "__main__":
    polish_dataset(INPUT_FILE, OUTPUT_FILE, BATCH_SIZE, MAX_INPUT_CHARS)
