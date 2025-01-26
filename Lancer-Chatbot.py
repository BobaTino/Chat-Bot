import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import docx
from transformers import pipeline
from collections import defaultdict
import spacy

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Lazy load heavy models
def get_spacy_model():
    return spacy.load("en_core_web_sm")

def get_semantic_similarity_model():
    return pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Function to extract text from a Word (.docx) file
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading Word document: {e}")
        return ""

# Pre-tokenize and store synonyms for the handbook
def preprocess_handbook(handbook_text):
    handbook_words = set(word_tokenize(handbook_text.lower()))
    synonym_dict = defaultdict(list)
    
    for word in handbook_words:
        synonyms = get_synonyms(word)
        synonym_dict[word] = synonyms
    return synonym_dict

# Get synonyms for a word
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Tokenize the user query
def tokenize_query(query):
    return word_tokenize(query.lower())

# Search function using tokens and synonyms
def search_with_tokens_and_synonyms(query_tokens, handbook_text, synonym_dict):
    handbook_text_lower = handbook_text.lower()

    for token in query_tokens:
        # Check for the token itself in the handbook
        if token in handbook_text_lower:
            start_index = handbook_text_lower.find(token)
            snippet = handbook_text[max(0, start_index - 50):start_index + 300]
            return f"Found relevant information: \n...\n{snippet}\n..."

        # Check for synonyms of the token in the handbook
        synonyms = synonym_dict.get(token, [])
        for synonym in synonyms:
            if synonym in handbook_text_lower:
                start_index = handbook_text_lower.find(synonym)
                snippet = handbook_text[max(0, start_index - 50):start_index + 300]
                return f"Found relevant information using synonym '{synonym}': \n...\n{snippet}\n..."

    return "Sorry, I can't find that information in the handbook."

# Enhanced NLP function as fallback
def search_with_advanced_nlp(query, handbook_text, semantic_similarity_model):
    # Lazy load spaCy and transformer models
    nlp = get_spacy_model()
    model = semantic_similarity_model

    # Extract entities and embeddings
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]  # Extract entities
    embeddings = model(query)[0]  # Generate sentence embeddings

    # Split the handbook text into paragraphs
    paragraphs = handbook_text.split("\n")
    best_match = None
    best_score = float("-inf")

    # Compute the average embedding for the query
    query_embedding_mean = [sum(x) / len(x) for x in zip(*embeddings)]

    for paragraph in paragraphs:
        paragraph_embedding = model(paragraph)[0]
        paragraph_embedding_mean = [sum(x) / len(x) for x in zip(*paragraph_embedding)]

        # Compute cosine similarity
        score = sum(a * b for a, b in zip(query_embedding_mean, paragraph_embedding_mean))

        if score > best_score:
            best_match = paragraph
            best_score = score

    return f"Found relevant information: \n...\n{best_match}\n..." if best_match else "Sorry, no relevant information found."

# Improved context-aware chatbot class
class ContextAwareChatbot:
    def __init__(self, handbook_text):
        self.previous_queries = []
        self.handbook_text = handbook_text
        self.synonym_dict = preprocess_handbook(handbook_text)
        self.semantic_similarity_model = None

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
            if "Sorry" in response:
                print("Chatbot: Let me perform a deeper search...")
                if not self.semantic_similarity_model:
                    self.semantic_similarity_model = get_semantic_similarity_model()
                response = search_with_advanced_nlp(user_input, self.handbook_text, self.semantic_similarity_model)

            print(f"Chatbot: {response}")

            # Store the query for context
            self.previous_queries.append(user_input)

# Main program execution
if __name__ == "__main__":
    # Specify the path to the Word (.docx) file
    file_path = "Text\IRB Handbook 3.3_FINAL.docx"

    # Extract text from the specified .docx file
    handbook_text = extract_text_from_docx(file_path)

    if handbook_text:
        # Initialize the context-aware chatbot and run it
        bot = ContextAwareChatbot(handbook_text)
        bot.chatbot()
    else:
        print("Error: Could not read the handbook.")
