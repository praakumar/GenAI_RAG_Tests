from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import json
from datetime import datetime, timedelta
from gmat_scraper import GmatScraper
import threading
import logging

app = Flask(__name__)

# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your API key

# Global variables
last_scrape_time = None
scrape_interval = timedelta(hours=24)  # Scrape new questions every 24 hours
qa_system = None
questions_lock = threading.Lock()

def load_questions():
    """Load questions from the JSON file."""
    try:
        with open('gmat_questions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_questions(questions):
    """Save questions to the JSON file."""
    with open('gmat_questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

def scrape_new_questions():
    """Scrape new questions and update the dataset."""
    global last_scrape_time
    
    try:
        logging.info("Starting to scrape new questions...")
        scraper = GmatScraper(use_selenium=True)
        scraper.scrape_all_sources()
        
        with questions_lock:
            # Load existing questions
            existing_questions = load_questions()
            
            # Add new questions
            new_questions = [q.__dict__ for q in scraper.questions]
            
            # Combine questions, avoiding duplicates based on question_text
            existing_texts = {q['question_text'] for q in existing_questions}
            unique_new_questions = [q for q in new_questions if q['question_text'] not in existing_texts]
            
            all_questions = existing_questions + unique_new_questions
            
            # Save updated questions
            save_questions(all_questions)
            
            # Update last scrape time
            last_scrape_time = datetime.now()
            
            logging.info(f"Added {len(unique_new_questions)} new questions to the dataset")
            
            # Reinitialize QA system with updated dataset
            initialize_qa_system()
            
    except Exception as e:
        logging.error(f"Error during question scraping: {str(e)}")
    finally:
        if scraper and scraper.driver:
            scraper.driver.quit()

def should_scrape():
    """Determine if it's time to scrape new questions."""
    if last_scrape_time is None:
        return True
    return datetime.now() - last_scrape_time > scrape_interval

def questions_to_text(questions):
    """Convert questions to a text format for the RAG system."""
    text = []
    for q in questions:
        text.append(f"Question: {q['question_text']}\n")
        text.append("Options:\n")
        for i, opt in enumerate(q['options'], 1):
            text.append(f"{i}. {opt}\n")
        text.append(f"Correct Answer: {q['correct_answer']}\n")
        text.append(f"Explanation: {q['explanation']}\n")
        text.append(f"Category: {q['category']}\n")
        text.append(f"Sub-category: {q['sub_category']}\n")
        text.append(f"Difficulty: {q['difficulty']}\n")
        text.append("-" * 80 + "\n")
    return "".join(text)

def initialize_qa_system():
    """Initialize or reinitialize the QA system with current questions."""
    global qa_system
    
    try:
        # Load questions
        questions = load_questions()
        
        # Convert questions to text format
        questions_text = questions_to_text(questions)
        
        # Save to temporary file for TextLoader
        with open('temp_questions.txt', 'w', encoding='utf-8') as f:
            f.write(questions_text)
        
        # Initialize QA system components
        loader = TextLoader('temp_questions.txt')
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        qa_system = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        # Clean up temporary file
        os.remove('temp_questions.txt')
        
    except Exception as e:
        logging.error(f"Error initializing QA system: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    # Check if we should scrape new questions
    if should_scrape():
        threading.Thread(target=scrape_new_questions).start()
    
    data = request.json
    question = data.get('question', '')
    
    try:
        # Get response from QA system
        response = qa_system.run(question)
        return jsonify({'answer': response, 'status': 'success'})
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        return jsonify({'answer': 'An error occurred while processing your question.', 'status': 'error'})

# Create templates directory and index.html
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
    <title>GMAT Question Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .question-box {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        #response {
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f0f0;
        }
        .loading {
            display: none;
            color: #666;
            font-style: italic;
        }
        .error {
            color: red;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>GMAT Question Generator</h1>
    <div class="question-box">
        <button onclick="getQuestion()">Generate New Question</button>
        <div id="loading" class="loading">Processing your request...</div>
        <div id="response"></div>
    </div>

    <script>
        function getQuestion() {
            const loading = document.getElementById('loading');
            const response = document.getElementById('response');
            
            loading.style.display = 'block';
            response.innerHTML = '';
            
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: "Generate a new GMAT question with its solution"
                })
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.status === 'success') {
                    response.innerHTML = data.answer;
                } else {
                    response.innerHTML = `<div class="error">${data.answer}</div>`;
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                response.innerHTML = '<div class="error">An error occurred while fetching the question.</div>';
            });
        }
    </script>
</body>
</html>
""")

if __name__ == '__main__':
    # Initialize the QA system before starting the server
    initialize_qa_system()
    app.run(debug=True)
