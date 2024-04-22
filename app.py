# Your existing imports
# Flask Imports
from flask import Flask, jsonify, render_template, request, redirect

# chatapp.py imports
import requests
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.anyscale import AnyscaleEmbedding
from llama_index.llms.anyscale import Anyscale
from config import ANYPONT_API_KEY, PUBMED_API_KEY
from appfunctions import fetch_data_from_pubmed, initialize_query_engine, predict_heart_disease

# Flask setup
app = Flask(__name__)

# initialize query engine
query_engine = initialize_query_engine()

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

@app.route("/submit_query", methods=["POST"])
def submit_query():
    # Get the query from the form data
    query = request.form.get("query")

    if query is None:
        # Return an error message if query is missing
        return "Invalid query"

    # Perform a query and get the response
    try:
        response = query_engine.query(query)
        # Render a new template with the response
        return render_template('submit_query.html', response=response)
    except Exception as e:
        # Return an error message if an error occurs
        return "Error: " + str(e)

# Example input data (replace with actual input)
@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        'Age': float(request.form["Age"]),
        'Sex': float(request.form["Sex"]),  # Example: 1 for male, 0 for female
        'ChestPainType': float(request.form["ChestPainType"]),
        'RestingBP': float(request.form["RestingBP"]),
        'Cholesterol': float(request.form["Cholesterol"]),
        'FastingBS': float(request.form["FastingBS"]),  # Example: 0 for false, 1 for true
        'RestingECG': float(request.form["RestingECG"]),
        'MaxHR': float(request.form["MaxHR"]),
        'ExerciseAngina': float(request.form["ExerciseAngina"]),  # Example: 0 for false, 1 for true
        'Oldpeak': float(request.form["Oldpeak"]),
        'ST_Slope': float(request.form["ST_Slope"])
    }

    # Make a prediction
    prediction = predict_heart_disease(input_data)

    # Render the result on a template
    return render_template("prediction_result.html", prediction=prediction)


# Add your existing routes here

if __name__ == '__main__':
    app.run(debug=True)
