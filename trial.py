import os
import json

# This is your folder where all JSON and PDF files are stored
base_folder = r"C:\Users\Kushal\Desktop\new\data\court_36_29_jan"

# Loop through all files in that folder
for file_name in os.listdir(base_folder):
    if file_name.endswith(".json"):
        json_path = os.path.join(base_folder, file_name)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract filename only from pdf_link
        pdf_link = data.get("pdf_link", "")
        pdf_filename = os.path.basename(pdf_link)  # Gives: HBHC010000022025_1_2025-01-22.pdf

        # Build full path to local PDF
        local_pdf_path = os.path.join(base_folder, pdf_filename)

        # Check if the PDF exists
        if os.path.exists(local_pdf_path):
            print(f"✅ Found PDF for JSON {file_name}: {pdf_filename}")
            # Do something with PDF
        else:
            print(f"❌ Missing PDF: {pdf_filename} for JSON {file_name}")
