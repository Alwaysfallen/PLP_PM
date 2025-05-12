import re
import PyPDF2
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

def extract_text_from_file(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_courseware_text(text):
    # 去除页码、版权、学校名等信息
    text = re.sub(r'\bPage \d+ of \d+\b', '', text)
    text = re.sub(r'\b(Copyright|©|All Rights Reserved).*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(University|Institute|School).*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(COPYRIGHT|CONFIDENTIAL|DRAFT|VERSION \d+)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Figure\s*\d+:.*', '', text, flags=re.IGNORECASE)

    # 替换公式为标记
    text = re.sub(r'(\$.*?\$|\\\[.*?\\\])', '[FORMULA]', text)

    # 去除奇怪的 Unicode 字符（保留英文、标点）
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 去除典型“表格标题 + 数值行”结构
    text = re.sub(r'(\d{1,3}(?:,\d{3})*\.?\d*\s+[A-Za-z ]+\s+(?:-?\d{1,3}(?:,\d{3})*\.?\d*\s*){2,})', '', text)

    # 删除数字密度过高的段落（数字比例 > 50%）
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        num_tokens = sum(1 for tok in tokens if re.match(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$', tok))
        if num_tokens / len(tokens) < 0.5:
            filtered_lines.append(line.strip())

    # 去重 + 去空行
    unique_lines = list(dict.fromkeys([l for l in filtered_lines if l]))
    text = '\n'.join(unique_lines)

    return text.strip()

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

def hybrid_keyword_extraction(clean_text, num_keywords=15):
    """Complete keyword extraction pipeline."""
    
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