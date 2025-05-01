import os
import sys
from cui_src.classify import classify_intent
from cui_src.preprocess import extract_text_from_pdf

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
        
        print("\nWelcome to the Intent Classification CUI!")
        print("Please upload a PDF document to begin.")
        print("Type 'exit' to quit.\n")

    def ask_for_document(self):
        """Prompt user for a PDF file and preprocess it"""
        while True:
            pdf_path = input("Enter the path of the PDF file: ").strip()
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
        # self.ask_for_document()

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
            summary = self.summarize_text(self.document_text)
            print(f"\n📄 Summary:\n{summary}")
        
        elif self.predicted_intent == "Ranked Keyword Extraction":
            keywords = self.extract_keywords(self.document_text)
            print(f"\n🔑 Keywords:\n{', '.join(keywords)}")

        elif self.predicted_intent == "Mind Map Visualization":
            print("\n🗺 Generating mind map... (Placeholder, integrate visualization logic here)")

        elif self.predicted_intent == "Multi-Document Consolidation":
            print("\n📑 Consolidating multiple documents... (Placeholder for processing logic)")

    def summarize_text(self, text):
        """Placeholder function for summarization"""
        return text[:500] + "..." 

    def extract_keywords(self, text):
        """Placeholder function for keyword extraction"""
        return ["keyword1", "keyword2", "keyword3"] 
    
    def exit_cui(self):
        """Handle graceful exit"""
        print("\nGoodbye! Exiting CUI...")
        sys.exit()

if __name__ == "__main__":
    cui = IntentCUI()
    cui.run()