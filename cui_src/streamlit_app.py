import streamlit as st
import os
from cui_src.classify import classify_intent
from kwe_src.mindmap1 import extract_text_from_pdf, clean_courseware_text
from keybert import KeyBERT
from transformers import pipeline, AutoTokenizer
import nltk
from nltk import sent_tokenize
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

# Initialize Models
kw_model = KeyBERT()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Streamlit UI
st.title("Conversational Intent-Based PDF Processor")
st.sidebar.write("Upload a PDF, and I will analyze it!")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file:
    st.success("✅ Document uploaded successfully!")

    # Extract text
    document_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_courseware_text(document_text)

    # User Input for Intent Classification
    user_query = st.text_input("Enter your query (Summarize, Keywords, Mind Map, etc.)")

    if user_query:
        predicted_intent, relevant_intents, high_confidence_intents = classify_intent(user_query)
        st.write(f"🔹 Predicted Intent: **{predicted_intent}**")

        if predicted_intent == "Document Summarization":
            summary = summarizer(cleaned_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.subheader("📄 Summary")
            st.write(summary)

        elif predicted_intent == "Ranked Keyword Extraction":
            keywords = kw_model.extract_keywords(cleaned_text, keyphrase_ngram_range=(1, 2), top_n=15)
            keywords_list = [kw for kw, _ in keywords]
            st.subheader("🔑 Keywords")
            st.write(", ".join(keywords_list))

        elif predicted_intent == "Mind Map Visualization":
            st.subheader("🗺 Mind Map Generation")
            input_ids = tokenizer(cleaned_text, truncation=True, max_length=None)['input_ids']

            # Summarization Handling
            if len(input_ids) > 1024:
                sentences = sent_tokenize(cleaned_text)
                chunks = []
                for sentence in sentences:
                    if len(tokenizer.tokenize(" ".join(chunks) + sentence)) <= 1024:
                        chunks.append(sentence)
                    else:
                        break

                summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                             for chunk in chunks]
                keywords = [kw for chunk in chunks for kw in kw_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 2), top_n=10)]
                summary = " ".join(summaries)
            else:
                summary = summarizer(cleaned_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                keywords = kw_model.extract_keywords(cleaned_text, keyphrase_ngram_range=(1, 2), top_n=10)

            # Build Mind Map Graph
            G = nx.DiGraph()
            keyword_weights = Counter([kw for kw, _ in keywords])
            central_topic = keyword_weights.most_common(1)[0][0]
            G.add_node("Root", label=central_topic)
            topics = sent_tokenize(summary)

            for topic in topics:
                G.add_node(topic, label=topic)
                G.add_edge("Root", topic)

            for keyword, _ in keywords:
                if keyword != central_topic and keyword not in topics:
                    G.add_node(keyword, label=keyword)
                    G.add_edge(topics[0], keyword)

            # Visualize and Save Mind Map
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            plt.figure(figsize=(10, 8))
            nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500, font_size=10)
            plt.title("Mind Map - Generated from PDF")
            plt.savefig("mind_map.png")
            st.image("mind_map.png")

            st.success("✅ Mind Map generated successfully!")
