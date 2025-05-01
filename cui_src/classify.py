import os
import re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from data import intent_examples

os.environ["TRANSFORMERS_NO_TF_IMPORT"] = "1"

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

SEMANTIC_THRESHOLD = 0.8  # Minimum confidence for sentence similarity
ZERO_SHOT_THRESHOLD = 0.5  # Minimum confidence for zero-shot classification

def preprocess_input(user_input):
    """Preprocess user input by lowercasing and removing punctuation."""
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', '', user_input)
    return user_input

def get_closest_intent(user_input):
    """Match user input to predefined intents using semantic similarity."""
    user_input = preprocess_input(user_input)
    user_embedding = semantic_model.encode(user_input)
    
    highest_score = 0
    best_match_intent = "Unknown"

    for intent, examples in intent_examples.items():
        example_embeddings = semantic_model.encode(examples)
        similarity_scores = util.cos_sim(user_embedding, example_embeddings)
        max_score = max(similarity_scores.tolist()[0])  # Get best match score
        
        if max_score > highest_score:
            highest_score = max_score
            best_match_intent = intent

    return best_match_intent, highest_score

def classify_intent(user_input):
    """Classify user intent using both semantic embeddings and zero-shot classification."""
    user_embedding = semantic_model.encode(user_input, convert_to_tensor=True)

    # Predefined intent embeddings
    intent_names = list(intent_examples.keys())
    intent_embeddings = semantic_model.encode(intent_names, convert_to_tensor=True)

    similarity_scores = util.cos_sim(user_embedding, intent_embeddings)

    best_match_index = similarity_scores.argmax().item()  # Index of highest similarity
    best_match_intent = intent_names[best_match_index]
    best_match_score = similarity_scores[0][best_match_index].item()

    if best_match_score >= SEMANTIC_THRESHOLD:
        return best_match_intent, [best_match_intent], [best_match_intent]

    result = classifier(user_input, candidate_labels=intent_names)

    predicted_intent = result["labels"][0]
    confidence_score = result["scores"][0]

    relevant_intents = [label for label, score in zip(result["labels"], result["scores"]) if score > ZERO_SHOT_THRESHOLD]
    high_confidence_intents = [label for label, score in zip(result["labels"], result["scores"]) if score > SEMANTIC_THRESHOLD]

    return predicted_intent, relevant_intents, high_confidence_intents