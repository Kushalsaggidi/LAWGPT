import os
import json
import pdfplumber
import requests
import re

# ---------- CONFIG ----------
BASE_DIR = r"C:\Users\Kushal\Desktop\new\data\court_36_29_jan"  # Directory with .json meta files
PDF_BASE_PATH = BASE_DIR                        # PDFs are stored here
PROGRESS_FILE = "progress.json"
CIVIL_OUTPUT = "civil_cases.json"
CRIMINAL_OUTPUT = "criminal_cases.json"
BATCH_SIZE = 5

GEMINI_API_KEY = "AIzaSyByofJwHz-ULKTZP3TboSnDSgxgTIrA9OM"
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
    try:
        cases_json = json.dumps(batch, ensure_ascii=False)[:14000]  # Truncate to fit token limit
    except Exception as e:
        print(f"Error serializing batch: {e}")
        return []
    prompt= """

    You are a legal document analyzer. For each case, read the context and determine whether it's a *civil* or *criminal* judgment.

    Based on the type, extract the required sections and return the output in the following JSON structure per case:

    {
    "instruction": "Extract the appropriate legal sections from this judgment.",
    "question": "Parse the judgment text of '<case_number> of <petitioner> Vs <respondent>'.",
    "answer": {
        "case_type": "Civil" or "Criminal",
        // if Civil:
        "contract_validity": "...",
        "property_rights": "...",
        "civil_procedure": "...",
        "court_direction": "...",
        "judgment_result": "..."
        
        // if Criminal:
        "criminal_liability_analysis": "...",
        "evidence_evaluation": "...",
        "sentencing_considerations": "...",
        "procedural_compliance": "...",
        "judgment_result": "..."
    }
    }

    Context and metadata for each case is provided below.

    Cases: {cases_json}
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
            if isinstance(ec, dict):
                case_type = ec.get("answer", {}).get("case_type", "").lower()
                if case_type == "civil":
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

