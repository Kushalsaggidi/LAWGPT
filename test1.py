import os
import fitz  # PyMuPDF

PDF_FOLDER = r"C:\\Users\\Kushal\\Desktop\\new\\2025_pdfs\\court=10_8\\bench=patnahcucisdb94"
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

print(f"Found {len(pdf_files)} PDF(s):")
print(pdf_files)

# Loop through and read each PDF
for filename in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, filename)
    print(f"\nReading: {filename}")
    
    # Open PDF
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()

    print(full_text[:1000])  # Show first 1000 characters (to avoid huge prints)
