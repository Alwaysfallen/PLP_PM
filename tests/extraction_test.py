from kwe_src.extraction import hybrid_keyword_extraction

pdf_path = r"C:\NUS ISS MTech in IS\EBA5004\Text Analytics (TA)\Day 3\S05 Information_Extraction_V5.3.pdf"  # Replace with actual PDF path
keywords = hybrid_keyword_extraction(pdf_path)
print("Extracted Keywords:", keywords)