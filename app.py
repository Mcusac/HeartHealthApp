# Your existing imports
# Flask Imports
from flask import Flask, jsonify, render_template, request, redirect

# chatapp.py imports
import requests
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.anyscale import AnyscaleEmbedding
from llama_index.llms.anyscale import Anyscale
from config import ANYPONT_API_KEY, PUBMED_API_KEY

# Flask setup
app = Flask(__name__)

# Routes
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/chat")
def chat():
    return render_template('chat.html')

@app.route("/predictor")
def predictor():
    return render_template('predictor.html')

# Add your existing routes here

if __name__ == '__main__':
    app.run(debug=True)
