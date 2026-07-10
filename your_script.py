import os
import json
import time
import wget
import zipfile
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import openai

# Get environment variables
ELASTIC_HOST = os.environ.get('ELASTIC_HOST', 'localhost')
ELASTIC_PORT = int(os.environ.get('ELASTIC_PORT', 9200))
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_CONTEXT_CHARS = int(os.environ.get('OPENAI_CONTEXT_CHARS', 3000))
SEARCH_RESULT_PREVIEW_CHARS = int(os.environ.get('SEARCH_RESULT_PREVIEW_CHARS', 500))

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

# Connect to Elasticsearch
# client = Elasticsearch(
#     [{'host': ELASTIC_HOST, 'port': ELASTIC_PORT}],
#     # If security is enabled, provide username and password
#     # http_auth=(os.environ.get('ELASTIC_USERNAME'), os.environ.get('ELASTIC_PASSWORD'))
# )
client = Elasticsearch(
    [{'host': ELASTIC_HOST, 'port': ELASTIC_PORT, 'scheme': 'http'}],
)

# Test connection
for attempt in range(1, 31):
    if client.ping():
        print("Connected to Elasticsearch")
        break
    print(f"Waiting for Elasticsearch at {ELASTIC_HOST}:{ELASTIC_PORT} ({attempt}/30)")
    time.sleep(2)
else:
    print("Could not connect to Elasticsearch")
    exit(1)

def truncate_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

# Download the dataset
def download_dataset():
    embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
    filename = 'vector_database_wikipedia_articles_embedded.zip'
    if not os.path.exists(filename):
        print("Downloading dataset...")
        wget.download(embeddings_url, filename)
    else:
        print("Dataset already downloaded.")
    # Extract the zip file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("Dataset extracted.")

# Create index with mapping
def create_index():
    index_mapping = {
        "properties": {
            "title_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },
            "text": {"type": "text"},
            "title": {"type": "text"},
            "url": {"type": "keyword"},
            "vector_id": {"type": "long"}
        }
    }
    if not client.indices.exists(index="wikipedia_vector_index"):
        client.indices.create(index="wikipedia_vector_index", mappings=index_mapping)
        print("Index created.")
    else:
        print("Index already exists.")

# Index data into Elasticsearch
def index_data():
    wikipedia_dataframe = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv")

    def dataframe_to_bulk_actions(df):
        for index, row in df.iterrows():
            yield {
                "_index": 'wikipedia_vector_index',
                "_id": row['id'],
                "_source": {
                    'url': row["url"],
                    'title': row["title"],
                    'text': row["text"],
                    'title_vector': json.loads(row["title_vector"]),
                    'content_vector': json.loads(row["content_vector"]),
                    'vector_id': row["vector_id"]
                }
            }

    print("Indexing data...")
    start = 0
    end = len(wikipedia_dataframe)
    batch_size = 100
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        batch_dataframe = wikipedia_dataframe.iloc[batch_start:batch_end]
        actions = dataframe_to_bulk_actions(batch_dataframe)
        helpers.bulk(client, actions)
    print("Data indexed.")

# Function to pretty print Elasticsearch results
def pretty_response(response):
    for hit in response['hits']['hits']:
        id = hit['_id']
        score = hit['_score']
        title = hit['_source']['title']
        text = truncate_text(hit['_source']['text'], SEARCH_RESULT_PREVIEW_CHARS)
        pretty_output = (f"\nID: {id}\nTitle: {title}\nSummary: {text}\nScore: {score}")
        print(pretty_output)

# Main function
def main():
    download_dataset()
    create_index()
    index_data()

    # Define question
    question = 'Is the Atlantic the biggest ocean in the world?'

    # Create embedding
    print("Generating question embedding...")
    question_embedding = openai.Embedding.create(input=question, model=EMBEDDING_MODEL)

    # Perform kNN search
    print("Performing kNN search...")
    response = client.search(
        index="wikipedia_vector_index",
        knn={
            "field": "content_vector",
            "query_vector": question_embedding["data"][0]["embedding"],
            "k": 10,
            "num_candidates": 100
        }
    )
    pretty_response(response)
    top_hit_text = response['hits']['hits'][0]['_source']['text']
    top_hit_summary = truncate_text(top_hit_text, OPENAI_CONTEXT_CHARS)
    print(f"Using {len(top_hit_summary)} of {len(top_hit_text)} characters from top search hit for OpenAI context.")

    # Use Chat Completion API
    print("Generating answer using OpenAI's Chat Completion API...")
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Answer the following question: "
                + question
                + " by using the following text: "
                + top_hit_summary,
            },
        ]
    )

    choices = summary.choices
    for choice in choices:
        print("------------------------------------------------------------")
        print(choice.message.content)
        print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
