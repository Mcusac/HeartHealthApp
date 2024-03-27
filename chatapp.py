# Imports
import requests
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.anyscale import AnyscaleEmbedding
from llama_index.llms.anyscale import Anyscale
from config import ANYPONT_API_KEY, PUBMED_API_KEY

PM_api_key = PUBMED_API_KEY

# Function to fetch data from the PubMed API
def fetch_data_from_pubmed(query):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "retmode": "json",
        "term": query,
        'api_key': PM_api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data from the PubMed API. Status code: {response.status_code}")
        return None

# Query for PubMed data
pubmed_query = "heart failure"
pubmed_data = fetch_data_from_pubmed(pubmed_query)

# Create document
document = Document(text=str(pubmed_data) if pubmed_data else "")

# Anypoint API key for LLM and embedding models
anypoint_api_key = ANYPONT_API_KEY

# Initialize LLM and embedding models
llm = Anyscale(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    api_key=anypoint_api_key
)

embed_model = AnyscaleEmbedding(
    model="thenlper/gte-large",
    api_key=anypoint_api_key,
    embed_batch_size=10
)

# Set settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Create index
index = VectorStoreIndex.from_documents([document])

query_engine = index.as_query_engine()

# Perform a query
response = query_engine.query(
    "I eat a lot of spinach. Will that help reduce heart issues?"
)
print(str(response))
