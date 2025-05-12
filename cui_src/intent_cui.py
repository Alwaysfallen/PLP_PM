import os
import sys
from data.intent_dataset import intent_examples
from cui_src.classify import classify_intent
from kwe_src.mindmap1 import extract_text_from_pdf, clean_courseware_text
from keybert import KeyBERT
from transformers import pipeline, AutoTokenizer
import nltk
from nltk import sent_tokenize
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

class IntentCUI:
    """
    Conversational UI (CUI) for Intent Classification.
    - First, prompts user to upload a PDF document.
    - Preprocesses the document (extracts text).
    - Then, classifies intent and performs corresponding action.
    """

    def __init__(self):
        """Initialize instance variables and welcome message"""
        self.user_input = ""
        self.predicted_intent = "Unknown"
        self.relevant_intents = []
        self.high_confidence_intents = []
        self.document_text = ""
        # Initialize KeyBERT model
        try:
            self.kw_model = KeyBERT()
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.kw_model = None


        print("\nWelcome to the Intent Classification CUI!")
        print("Please upload a PDF document to begin.")
        print("Type 'exit' to quit.\n")

    def ask_for_document(self):
        """Prompt user for a PDF file and preprocess it"""
        while True:
            pdf_path = r"{}".format(input("Enter the path of the PDF file: ").strip())
            if pdf_path.lower() == "exit":
                self.exit_cui()

            if os.path.exists(pdf_path) and pdf_path.endswith(".pdf"):
                self.document_text = extract_text_from_pdf(pdf_path)
                print("\n✅ Document successfully processed!\n")
                break
            else:
                print("⚠️ Invalid file path or format. Please provide a valid PDF.")



    def run(self):
        """Start the CUI loop"""
        self.ask_for_document()

        while True:
            self.user_input = input("Enter your query: ").strip()

            # Classify intent and update instance variables
            self.predicted_intent, self.relevant_intents, self.high_confidence_intents = classify_intent(self.user_input)

            if self.predicted_intent == "Exit Application":
                print("\nGoodbye! Exiting CUI...")
                sys.exit()

            self.perform_action()

    def perform_action(self):
        """Perform action based on classified intent"""
        print(f"\n🔹 Predicted Intent: {self.predicted_intent}")

        if self.predicted_intent == "Document Summarization":
            if not self.document_text:
                print("⚠️ No document loaded. Please upload a PDF first.")
                return
            summary = self.summarize_text(self.document_text)
            print(f"\n📄 Summary:\n{summary}")

        elif self.predicted_intent == "Ranked Keyword Extraction":
            if not self.document_text:
                print("⚠️ No document loaded. Please upload a PDF first.")
                return
            if self.kw_model is None:
                 print("⚠️ Keyword extraction model not initialized properly.")
                 return
            keywords = self.extract_keywords(self.document_text)
            print(f"\n🔑 Keywords:\n{', '.join(keywords)}")

        elif self.predicted_intent == "Mind Map Visualization":
            if not self.document_text:
                print("⚠️ No document loaded. Please upload a PDF first.")
                return
    
            print("\n🗺 Generating mind map...")
            cleaned_text = clean_courseware_text(not self.document_text)
            image = self.generate_mind_map(cleaned_text)
            print("\n✅ Mind map generated and saved as 'mind_map.png'!")

        else:
            print("\n🤔 Sorry, I didn't understand that intent.")


    def summarize_text(self, text):
        """Placeholder function for summarization"""
        return text[:500] + "..." 

    def extract_keywords(self, text, num_keywords=15):
        """
        Extract keywords from text using KeyBERT.
        """
        if not text or self.kw_model is None:
            return []

        try:
            # keywords_with_scores = self.kw_model.extract_keywords(text, top_n=num_keywords)
            keywords_with_scores = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=num_keywords)
            # Extract just the keywords
            keywords = [keyword for keyword, score in keywords_with_scores]
            return keywords
        except Exception as e:
            print(f"An error occurred during KeyBERT keyword extraction: {e}")
            return []
        
    def generate_mind_map(self, text):
        input_ids = self.tokenizer(text, truncation=True, max_length=None)['input_ids']
        if len(input_ids) > 1024:
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = ""
            for sentence in sentences:
                if len(self.tokenizer.tokenize(current_chunk + sentence)) <= 1024:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk.strip())

            summaries = []
            keywords_list = []
            for chunk in chunks:
                summary = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                keywords = self.kw_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 2), top_n=10)
                summaries.append(summary)
                keywords_list.append(keywords)

            summary = " ".join(summaries)
            keywords = [kw for sublist in keywords_list for kw in sublist]

        else:
            summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=10)

        # Build mind map graph
        G = nx.DiGraph()
        keyword_weights = Counter([kw for kw, score in keywords])
        central_topic = keyword_weights.most_common(1)[0][0]
        G.add_node("Root", label=central_topic)

        topics = nltk.sent_tokenize(summary)
        for topic in topics:
            G.add_node(topic, label=topic)
            G.add_edge("Root", topic)

        for keyword, score in keywords:
            if keyword != central_topic and keyword not in topics:
                G.add_node(keyword, label=keyword)
                G.add_edge(topics[0], keyword)

        # Visualize mind map
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500, font_size=10)
        plt.title("Mind Map - Generated from PDF")
        plt.savefig("/logs/mind_map.png")
    
    def exit_cui(self):
        """Handle graceful exit"""
        print("\nGoodbye! Exiting CUI...")
        sys.exit()

if __name__ == "__main__":
    cui = IntentCUI()
    cui.run()