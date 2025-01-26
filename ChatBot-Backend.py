import openai
import numpy as np
import docx
import PyPDF2
import os

# Set your OpenAI API key securely
openai.api_key = ('API_KEY')  # Make sure to set this environment variable

# Extract text from a DOCX file
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

# Extract text based on the file type
def extract_guidebook_data(file_path):
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .docx or .pdf file.")

# Generate embeddings using OpenAI
def get_guidebook_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'])

# Compare query to guidebook content using embeddings
def get_relevant_section(query, guidebook_text, chunk_size=500):
    guidebook_chunks = [guidebook_text[i:i+chunk_size] for i in range(0, len(guidebook_text), chunk_size)]
    query_embedding = get_guidebook_embedding(query)
    
    # Compute similarity for each chunk
    chunk_embeddings = [get_guidebook_embedding(chunk) for chunk in guidebook_chunks]
    similarities = [np.dot(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
    
    # Get the chunk with the highest similarity
    most_relevant_chunk = guidebook_chunks[np.argmax(similarities)]
    return most_relevant_chunk

# Generate a response from OpenAI based on relevant guidebook section
def generate_response(query, guidebook_text):
    relevant_section = get_relevant_section(query, guidebook_text)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant using guidebook data."},
            {"role": "user", "content": f"Faculty question: {query}"},
            {"role": "assistant", "content": f"Relevant guidebook section: {relevant_section}"}
        ]
    )
    return response['choices'][0]['message']['content']

# CLI function to handle terminal input
def run_cli():
    print("Welcome to the Guidebook Q&A CLI.")
    
    # Ask the user for the file path
    file_path = input("Enter the path of the guidebook file (DOCX or PDF): ")
    
    # Extract the guidebook text
    try:
        guidebook_text = extract_guidebook_data(file_path)
        print("Guidebook data loaded successfully.")
    except Exception as e:
        print(f"Error loading guidebook: {e}")
        return
    
    # Enter the query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the Guidebook Q&A CLI.")
            break
        
        # Generate a response using the guidebook data
        try:
            response = generate_response(query, guidebook_text)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error generating response: {e}")

# Entry point for CLI or Flask app
if __name__ == "__main__":
    run_mode = input("Type 'cli' for terminal interaction or 'server' to start the Flask server: ").lower()

    if run_mode == "cli":
        run_cli()  # Run the CLI mode
    elif run_mode == "server":
        from flask import Flask, request, jsonify, render_template

        app = Flask(__name__)

        @app.route("/")
        def index():
            return render_template("index.html")  # Serve the HTML file

        @app.route("/chat", methods=["POST"])
        def chat():
            user_input = request.json.get("query")
            file_type = request.json.get("file_type")

            if not user_input or not file_type:
                return jsonify({"error": "No query or file type provided"}), 400

            guidebook_file_path = 'guidebook.docx'  # Adjust based on your actual setup
            guidebook_text = extract_guidebook_data(guidebook_file_path)
            response = generate_response(user_input, guidebook_text)
            return jsonify({"response": response})

        app.run(debug=True)  # Run Flask app if server mode is selected
    else:
        print("Invalid option. Please restart and choose either 'cli' or 'server'.")
