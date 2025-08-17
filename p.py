import os
import json
import pdfplumber
import requests
import re

# ---------- CONFIG ----------
BASE_DIR = "metadata-2025-taphc"  # Directory with .json meta files
PDF_BASE_PATH = BASE_DIR                        # PDFs are stored here
PROGRESS_FILE = "progress.json"
CIVIL_OUTPUT = "civil_cases.json"
CRIMINAL_OUTPUT = "criminal_cases.json"
BATCH_SIZE = 5

GEMINI_API_KEY = "AIzaSyCz41qZibYl1rca9Ye7_pkKaaBlQnms6ME"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"

# ---------- LOAD / SAVE PROGRESS ----------
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_processed": None}

def save_progress(last_file):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_processed": last_file}, f)

# ---------- EXTRACT PDF TEXT ----------
def extract_pdf_text(pdf_path):
    if not os.path.exists(pdf_path):
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

# ---------- GEMINI REQUEST ----------
def query_gemini(batch):
    prompt = f"""
    You are a legal document analyzer. For each case, return ONLY valid JSON with fields:
    {{
        "case_type": "Civil" or "Criminal",
        "court_findings": "Summary of key findings (fill appropriately if missing)",
        "court_reasoning": "Brief reasoning behind judgment (fill appropriately if missing)",
        "judgment_result": "Final result of the case (predict based on context if missing)"
    }}

    Respond as a JSON array of enriched cases.

    Cases: {json.dumps(batch)[:14000]}
    """
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_URL, json=payload)
        result = response.json()

        gemini_text = result["candidates"][0]["content"]["parts"][0]["text"]
        match = re.search(r"\[.*\]", gemini_text, re.DOTALL)
        clean_json = match.group(0) if match else gemini_text

        return json.loads(clean_json)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return []

# ---------- MAIN ----------
def main():
    all_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".json")])
    progress = load_progress()

    start_index = 0
    if progress["last_processed"] in all_files:
        start_index = all_files.index(progress["last_processed"]) + 1

    civil_cases = json.load(open(CIVIL_OUTPUT, "r", encoding="utf-8")) if os.path.exists(CIVIL_OUTPUT) else []
    criminal_cases = json.load(open(CRIMINAL_OUTPUT, "r", encoding="utf-8")) if os.path.exists(CRIMINAL_OUTPUT) else []

    for i in range(start_index, len(all_files), BATCH_SIZE):
        batch_files = all_files[i:i+BATCH_SIZE]
        batch = []

        for file_name in batch_files:
            case_data = json.load(open(os.path.join(BASE_DIR, file_name), "r", encoding="utf-8"))
            if isinstance(case_data, dict):
                case_data = [case_data]

            for case in case_data:
                pdf_filename = os.path.basename(case["pdf_link"])
                pdf_path = os.path.join(PDF_BASE_PATH, pdf_filename)
                case["pdf_text"] = extract_pdf_text(pdf_path)
                batch.append(case)

        enriched_cases = query_gemini(batch)

        for ec in enriched_cases:
            if ec.get("case_type", "").lower() == "civil":
                civil_cases.append(ec)
            else:
                criminal_cases.append(ec)

        json.dump(civil_cases, open(CIVIL_OUTPUT, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        json.dump(criminal_cases, open(CRIMINAL_OUTPUT, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        save_progress(batch_files[-1])

        print(f"Processed up to: {batch_files[-1]}")

    print("All cases processed.")

if __name__ == "__main__":
    main()
