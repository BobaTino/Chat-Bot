import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import docx
from transformers import pipeline as hf_pipeline
from collections import defaultdict
import spacy
from functools import lru_cache
import random
from pymongo import MongoClient
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize spaCy and NLTK components
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Cache the sentence-transformers model
@lru_cache(maxsize=1)
def get_semantic_similarity_model():
    return SentenceTransformer('all-mpnet-base-v2').to("cuda" if torch.cuda.is_available() else "cpu")

# Cache the summarization model
@lru_cache(maxsize=1)
def get_summarization_model():
    return hf_pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract text from a Word (.docx) file
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        print(f"Error reading Word document: {e}")
        return ""

# Preprocess text: tokenize, lemmatize, and remove stop words
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]

# Cache the preprocessed handbook and synonym dictionary
@lru_cache(maxsize=1)
def preprocess_handbook(handbook_text):
    handbook_words = set(preprocess_text(handbook_text))
    synonym_dict = defaultdict(list)
    
    for word in handbook_words:
        synonyms = get_synonyms(word)
        synonym_dict[word] = synonyms
    return synonym_dict

# Get synonyms for a word using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

# Tokenize and preprocess the user query
def tokenize_query(query):
    return preprocess_text(query)

# Summarize the given text with an option for random sampling
def summarize_text(text, max_length=70, do_sample=False):
    if len(text.split()) <= 10:  # Skip summarization for very short text
        return text
    
    summarizer = get_summarization_model()
    try:
        summary = summarizer(text, max_length=max_length, min_length=35, do_sample=do_sample)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Summarization failed: {e}")
        return text  # Fallback to original text if summarization fails

# Search function using tokens and synonyms
def search_with_tokens_and_synonyms(query_tokens, handbook_text, synonym_dict):
    handbook_text_lower = handbook_text.lower()

    for token in query_tokens:
        # Check for the token itself in the handbook
        if token in handbook_text_lower:
            start_index = handbook_text_lower.find(token)
            snippet = handbook_text[max(0, start_index - 50):start_index + 300]
            summary = summarize_text(snippet)
            return f"Found relevant information:\n{summary}\n"

        # Check for synonyms of the token in the handbook
        synonyms = synonym_dict.get(token, [])
        for synonym in synonyms:
            if synonym in handbook_text_lower:
                start_index = handbook_text_lower.find(synonym)
                snippet = handbook_text[max(0, start_index - 200):start_index + 600]
                summary = summarize_text(snippet)
                return f"Found relevant information using synonym '{synonym}': \n...\n{summary}\n..."

    return None  # No match found

# Enhanced NLP function with better semantic search and randomization option
def search_with_advanced_nlp(query, handbook_text, semantic_similarity_model, randomize=False):
    # Split the handbook text into paragraphs
    paragraphs = [p.strip() for p in handbook_text.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return "No relevant information found."
    
    # Compute query embedding
    query_embedding = semantic_similarity_model.encode(query, convert_to_tensor=True)
    # Compute embeddings for all paragraphs
    paragraph_embeddings = semantic_similarity_model.encode(paragraphs, convert_to_tensor=True)
    
    # Compute cosine similarity between query and paragraphs
    similarities = util.pytorch_cos_sim(query_embedding, paragraph_embeddings)[0]
    
    if randomize:
        # Get top 3 paragraphs and filter by a similarity threshold
        k = min(3, len(similarities))
        topk = torch.topk(similarities, k=k)
        indices = topk.indices.tolist()
        scores = topk.values.tolist()
        valid_indices = [idx for idx, score in zip(indices, scores) if score > 0.5]
        if valid_indices:
            selected_index = random.choice(valid_indices)
        else:
            selected_index = similarities.argmax().item()
        best_match = paragraphs[selected_index]
        # Use sampling in summarization to produce varied output
        summary = summarize_text(best_match, do_sample=True)
        return f"Found relevant information: \n...\n{summary}\n..."
    else:
        # Deterministically pick the best match
        best_index = similarities.argmax().item()
        best_match = paragraphs[best_index]
        best_score = similarities[best_index].item()
        if best_score > 0.5:  # Threshold for relevance
            summary = summarize_text(best_match)
            return f"Found relevant information: \n...\n{summary}\n..."
        else:
            return "Sorry, no relevant information found."

# Improved context-aware chatbot class with feedback loop for answer regeneration
class ContextAwareChatbot:
    def __init__(self, handbook_text):
        self.handbook_text = handbook_text
        self.synonym_dict = preprocess_handbook(handbook_text)
        self.semantic_similarity_model = get_semantic_similarity_model()
        
        # Connect to MongoDB (replace with your actual connection details)
        self.client = MongoClient("mongodb+srv://longle_user:longle_password@iq-cluster.dujj4.mongodb.net/?retryWrites=true&w=majority&appName=IQ-Cluster")
        self.db = self.client["chatbot_query"]
        self.collection = self.db["bot_info"]

    def chatbot(self):
        print("Chatbot: How can I assist you? Type 'bye' to exit.")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "bye":
                print("Chatbot: Goodbye!")
                break

            # Display previous queries for context
            previous_queries = self.collection.find().sort("_id", -1).limit(5)
            if previous_queries:
                print("Chatbot: Previously you asked about:")
                for query in previous_queries:
                    print(f"- {query['query']}")

            # Tokenize the query
            tokens = tokenize_query(user_input)

            # First attempt: Token and synonym search
            response = search_with_tokens_and_synonyms(tokens, self.handbook_text, self.synonym_dict)

            # If no match, fallback to semantic similarity
            if not response:
                print("Chatbot: Let me perform a deeper search...")
                response = search_with_advanced_nlp(user_input, self.handbook_text, self.semantic_similarity_model)

            # Feedback loop: ask if the answer is good
            while True:
                print(f"Chatbot: {response}")
                feedback = input("Was this answer helpful? (yes/no): ").strip().lower()
                if feedback == "yes":
                    break
                elif feedback == "no":
                    print("Chatbot: Let me try to provide a better answer.")
                    # Use randomized semantic search to generate a different answer
                    response = search_with_advanced_nlp(user_input, self.handbook_text, self.semantic_similarity_model, randomize=True)
                else:
                    print("Please answer with 'yes' or 'no'.")

            # Store the query and final response in MongoDB
            try: 
                document = {
                    "query": user_input,
                    "response": response,
                    "timestamp": datetime.utcnow()
                }
                print("Inserting document:", document)
                self.collection.insert_one(document)
                print("Document inserted successfully.")
            except Exception as e:
                print(f"Error inserting document into MongoDB: {e}")

# Main program execution
if __name__ == "__main__":
    # Specify the path to the Word (.docx) file
    file_path = "Text/IRB Handbook 3.3_FINAL.docx"

    # Extract text from the specified .docx file
    handbook_text = extract_text_from_docx(file_path)

    if handbook_text:
        # Initialize the context-aware chatbot and run it
        bot = ContextAwareChatbot(handbook_text)
        bot.chatbot()
    else:
        print("Error: Could not read the handbook.")
