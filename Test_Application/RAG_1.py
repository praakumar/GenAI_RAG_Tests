from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

app = Flask(__name__)

# Initialize OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Load and process GMAT questions dataset
def initialize_qa_system():
    # Load GMAT questions dataset
    loader = TextLoader('gmat_questions.txt')  # Create this file with GMAT questions
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain

qa_system = initialize_qa_system()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    
    # Get response from QA system
    response = qa_system.run(question)
    
    return jsonify({'answer': response})

# Create templates/index.html with this content
"""
<!DOCTYPE html>
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
    </style>
</head>
<body>
    <h1>GMAT Quantitative Question Generator</h1>
    <div class="question-box">
        <button onclick="getQuestion()">Generate New Question</button>
        <div id="response"></div>
    </div>

    <script>
        function getQuestion() {
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: "Generate a new GMAT quantitative question with its solution"
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('response').innerHTML = data.answer;
            });
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
