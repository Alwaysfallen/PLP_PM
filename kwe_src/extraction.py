import os
import re
import fitz  # PyMuPDF for PDF text extraction
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertForTokenClassification

# Load NLP models
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load SciBERT for technical term extraction
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = BertForTokenClassification.from_pretrained("allenai/scibert_scivocab_uncased")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF document."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def preprocess_text(text):
    """Enhance text cleaning while retaining key AI terms."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop or token.ent_type_])

def extract_keywords_keybert(text, num_keywords=10):
    """Extract keywords using KeyBERT."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    return [kw[0] for kw in keywords]

def extract_keywords_sciBERT(text, chunk_size=512):
    """Split large text into chunks and process separately."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Split tokenized input into smaller parts
    token_chunks = [tokens["input_ids"][0][i:i+chunk_size] for i in range(0, len(tokens["input_ids"][0]), chunk_size)]

    keywords = set()
    for chunk in token_chunks:
        outputs = model(input_ids=chunk.unsqueeze(0))
        predicted_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in outputs.logits.argmax(dim=-1).tolist()[0]]
        keywords.update(predicted_tokens)  # Merge keywords from all chunks

    return list(keywords)

def refine_keywords(keyword_list):
    """Refine keywords by clustering similar ones."""
    embeddings = embed_model.encode(keyword_list)
    clustered_keywords = util.community_detection(embeddings, threshold=0.7)
    return clustered_keywords

def hybrid_keyword_extraction(pdf_path, num_keywords=15):
    """Complete keyword extraction pipeline."""
    raw_text = extract_text_from_pdf(pdf_path)
    clean_text = preprocess_text(raw_text)
    
    keybert_keywords = extract_keywords_keybert(clean_text, num_keywords)
    # sciBERT_keywords = extract_keywords_sciBERT(clean_text)

    # combined_keywords = list(set(keybert_keywords + sciBERT_keywords))  # Merge results
    combined_keywords = list(set(keybert_keywords))  # Merge results
    refined_keywords = refine_keywords(combined_keywords)  # Apply semantic clustering

    return refined_keywords

# Example Usage
# pdf_path = "sample_courseware.pdf"  # Replace with actual PDF path
# keywords = hybrid_keyword_extraction(pdf_path)
# print("Extracted Keywords:", keywords)