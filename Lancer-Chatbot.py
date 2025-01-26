import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import docx
from transformers import pipeline
import spacy
import tensorflow as tf

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load a pre-trained sentence embedding model from transformers
semantic_similarity_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

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

# Tokenize the user input using NLTK
def tokenize_query(query):
    return word_tokenize(query.lower())

# Get synonyms for keywords in the query
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Enhanced NLP: Identify entities and generate embeddings for better matching
def extract_entities_and_embeddings(query):
    doc = nlp(query)
    entities = [ent.text for ent in doc.ents]  # Extract entities
    embeddings = semantic_similarity_model(query)[0]  # Generate sentence embeddings
    return entities, embeddings
    
# Search function that uses tokens and synonyms
def search_handbook_with_synonyms(tokens, handbook_text):
    handbook_text_lower = handbook_text.lower()
    
    for token in tokens:
        # Search for the token itself
        if token in handbook_text_lower:
            start_index = handbook_text_lower.find(token)
            snippet = handbook_text[max(0, start_index - 50):start_index + 300]
            return f"Found relevant information: \n...\n{snippet}\n..."
        
        # Search for synonyms of the token
        synonyms = get_synonyms(token)
        for synonym in synonyms:
            if synonym in handbook_text_lower:
                start_index = handbook_text_lower.find(synonym)
                snippet = handbook_text[max(0, start_index - 50):start_index + 300]
                return f"Found relevant information using synonym '{synonym}': \n...\n{snippet}\n..."
    
    return "Sorry, I can't find that information in the handbook."

# Search using semantic similarity and synonyms
def search_handbook_with_enhanced_nlp(query, handbook_text):
    # Get entities and embeddings for advanced search
    entities, query_embedding = extract_entities_and_embeddings(query)
    
    # Split the handbook text into paragraphs for better matching
    paragraphs = handbook_text.split("\n")
    best_match = None
    best_score = float("-inf")
    
    # Compute the average embedding for the query
    query_embedding_mean = [sum(x) / len(x) for x in zip(*query_embedding)]

    for paragraph in paragraphs:
        # Generate paragraph embeddings
        paragraph_embedding = semantic_similarity_model(paragraph)[0]
        
        # Compute cosine similarity
        score = sum(a * b for a, b in zip(query_embedding_mean, paragraph_embedding[0]))
        
        # Update best match if current score is higher
        if score > best_score:
            best_match = paragraph
            best_score = score
    
    # Fallback to synonym matching if semantic similarity finds no match
    if best_score < 0.7:  # Adjustable threshold
        tokens = tokenize_query(query)
        best_match = search_handbook_with_synonyms(tokens, handbook_text)  # Existing synonym function
    
    return f"Found relevant information: \n...\n{best_match}\n..." if best_match else "Sorry, no relevant information found."

# Improved context-aware chatbot class
class ContextAwareChatbot:
    def __init__(self):
        self.previous_queries = []

    def chatbot(self, handbook_text):
        print("Chatbot: How can I assist you? Type 'bye' to exit.")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "bye":
                print("Chatbot: Goodbye!")
                break
            
            # Display previous queries for context, if applicable
            if self.previous_queries:
                print(f"Chatbot: Previously you asked about '{self.previous_queries[-1]}'...")
            
            # Generate response with enhanced NLP
            response = search_handbook_with_enhanced_nlp(user_input, handbook_text)
            print(f"Chatbot: {response}")
            
            # Store the current query for context
            self.previous_queries.append(user_input)

# Main program execution
if __name__ == "__main__":
    # Specify the path to the Word (.docx) file
    file_path = "Text\IRB Handbook 3.3_FINAL.docx"
    
    # Extract text from the specified .docx file
    handbook_text = extract_text_from_docx(file_path)
    
    if handbook_text:
        # Initialize the context-aware chatbot and run it
        bot = ContextAwareChatbot()
        bot.chatbot(handbook_text)
    else:
        print("Error: Could not read the handbook.")