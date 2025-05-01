import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        text = ""

        # Iterate through each page and extract text
        for page in doc:
            text += page.get_text("text") + "\n"

        return text.strip()
    
    except Exception as e:
        print(f"⚠️ Error extracting text: {e}")
        return None