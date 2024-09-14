from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.chains import RetrievalQA
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from waitress import serve
import logging
import time

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "20 per hour"],
    storage_uri="memory://"
)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable not set")

llm = ChatCohere(cohere_api_key=COHERE_API_KEY, model="command", max_tokens=300, temperature=0.7)

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

try:
    db = Chroma(persist_directory="chroma_index", embedding_function=embeddings)
except Exception as e:
    print(f"Error initializing Chroma: {e}")
    print("Waiting 60 seconds before retrying...")
    time.sleep(60)
    db = Chroma(persist_directory="chroma_index", embedding_function=embeddings)

retv = db.as_retriever(search_kwargs={"k": 8})

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retv)

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=60))
def safe_embed(text):
    return embeddings.embed_query(text)

@lru_cache(maxsize=100)
def cached_chain_run(user_input):
    return chain.run(user_input)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@limiter.limit("5 per minute")
def chat():
    try:
        user_input = request.json.get('message')
        embedded_input = safe_embed(user_input)
        response = cached_chain_run(user_input)
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        if "429" in str(e):
            return jsonify({'response': "High demand issue. Please try again in a few moments."}), 429
        return jsonify({'response': "Trouble in processing the request right now. Please try again later."}), 500

@app.route('/test')
def test():
    return "Server is running!"

if __name__ == '__main__':
    try:
        print("Script is running...")
        sys.stdout.flush()
        logger = logging.getLogger('waitress')
        logger.setLevel(logging.INFO)
        print("Starting Waitress server...")
        sys.stdout.flush()
        print("Server is running on http://127.0.0.1:8080")
        sys.stdout.flush()
        serve(app, host='127.0.0.1', port=8080)
    except Exception as e:
        print(f"An error occurred while starting the server: {e}")
        import traceback
        traceback.print_exc()