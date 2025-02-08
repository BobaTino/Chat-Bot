import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import docx
from transformers import pipeline as hf_pipeline
from collections import defaultdict
import spacy
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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
    return hf_pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

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

# Summarize the given text
def summarize_text(text, max_length=150):
    if len(text.split()) <= 10:  # Skip summarization for very short text
        return text
    
    summarizer = get_summarization_model()
    try:
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
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
            return f"Found relevant information: \n...\n{summary}\n..."

        # Check for synonyms of the token in the handbook
        synonyms = synonym_dict.get(token, [])
        for synonym in synonyms:
            if synonym in handbook_text_lower:
                start_index = handbook_text_lower.find(synonym)
                snippet = handbook_text[max(0, start_index - 50):start_index + 300]
                summary = summarize_text(snippet)
                return f"Found relevant information using synonym '{synonym}': \n...\n{summary}\n..."

    return None  # No match found

# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Enhanced NLP function as fallback
def search_with_advanced_nlp(query, handbook_text, semantic_similarity_model):
    # Split the handbook text into paragraphs
    paragraphs = [p for p in handbook_text.split("\n") if p.strip()]
    
    # Compute query embedding
    query_embedding = np.mean(semantic_similarity_model(query)[0], axis=0)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        future_to_paragraph = {
            executor.submit(
                lambda p: (cosine_similarity(query_embedding, np.mean(semantic_similarity_model(p)[0], axis=0)), p),
                paragraph
            ): paragraph for paragraph in paragraphs
        }
        
        best_match, best_score = None, float("-inf")
        for future in future_to_paragraph:
            score, paragraph = future.result()
            if score > best_score:
                best_match, best_score = paragraph, score

    if best_match:
        summary = summarize_text(best_match)
        return f"Found relevant information: \n...\n{summary}\n..."
    else:
        return "Sorry, no relevant information found."

# Improved context-aware chatbot class
class ContextAwareChatbot:
    def __init__(self, handbook_text):
        self.previous_queries = []
        self.handbook_text = handbook_text
        self.synonym_dict = preprocess_handbook(handbook_text)
        self.semantic_similarity_model = get_semantic_similarity_model()

    def chatbot(self):
        print("Chatbot: How can I assist you? Type 'bye' to exit.")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() == "bye":
                print("Chatbot: Goodbye!")
                break

            # Display previous queries for context
            if self.previous_queries:
                print(f"Chatbot: Previously you asked about '{self.previous_queries[-1]}'...")

            # Tokenize the query
            tokens = tokenize_query(user_input)

            # First attempt: Token and synonym search
            response = search_with_tokens_and_synonyms(tokens, self.handbook_text, self.synonym_dict)

            # If no match, fallback to semantic similarity
            if not response:
                print("Chatbot: Let me perform a deeper search...")
                response = search_with_advanced_nlp(user_input, self.handbook_text, self.semantic_similarity_model)

            print(f"Chatbot: {response}")

            # Store the query for context
            self.previous_queries.append(user_input)

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