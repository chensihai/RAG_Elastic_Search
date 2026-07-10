Project Title
Retrieval Augmented Generation with Elasticsearch and OpenAI

Introduction
This project demonstrates a small Retrieval Augmented Generation (RAG) workflow
using Elasticsearch for vector search and OpenAI for embeddings and answer
generation.

The application does the following:

1. Starts Elasticsearch and Kibana with Docker Compose.
2. Downloads a pre-embedded Wikipedia dataset from the OpenAI examples CDN.
3. Creates an Elasticsearch vector index named wikipedia_vector_index.
4. Bulk indexes the Wikipedia article vectors.
5. Embeds a sample question with OpenAI.
6. Runs a k-NN search in Elasticsearch.
7. Sends the top retrieved text, capped to a configurable size, to OpenAI Chat
   Completion to generate an answer.

Components
Elasticsearch
Stores the article text, metadata, and dense vectors. The compose file runs
Elasticsearch 8.9.1 as a single-node local development cluster with security
disabled.

Kibana
Runs at http://localhost:5601 and can be used to inspect index status, document
counts, and search results.

Application
A Python script, your_script.py, handles downloading, indexing, search, and the
OpenAI calls.

Important Version Note
The Python Elasticsearch client must be compatible with Elasticsearch 8.x.
requirements.txt pins:

elasticsearch>=8.9,<9

Without this pin, pip may install elasticsearch-py 9.x. That client sends v9
compatibility headers, which Elasticsearch 8.9.1 rejects. The symptom is:

Could not connect to Elasticsearch

Prerequisites
Docker and Docker Compose installed.
An OpenAI API key.

Setup
Clone the repository:

git clone https://github.com/chensihai/RAG_Elastic_Search
cd RAG_Elastic_Search

Copy the sample environment file:

cp .env.sample .env

Edit .env and set your OpenAI API key:

OPENAI_API_KEY=sk-your-openai-api-key

The sample also includes optional limits:

OPENAI_CONTEXT_CHARS=3000
SEARCH_RESULT_PREVIEW_CHARS=500

OPENAI_CONTEXT_CHARS controls how much retrieved article text is sent to OpenAI.
This helps control token usage and cost.

SEARCH_RESULT_PREVIEW_CHARS controls how much retrieved text is printed in the
application logs.

Run
Build and start the stack:

docker compose up -d --build

Follow the app logs:

docker compose logs -f app

Expected progress:

Connected to Elasticsearch
Downloading dataset...
Dataset extracted.
Index created.
Indexing data...
Data indexed.
Generating question embedding...
Performing kNN search...
Using 3000 of ... characters from top search hit for OpenAI context.
Generating answer using OpenAI's Chat Completion API...

The app is a one-shot script. It is normal for the app container to exit after
the indexing, search, and OpenAI answer generation finish.

If you only want to recreate the app service after code changes:

docker compose up -d --build --force-recreate app

Monitor With Kibana
Open Kibana:

http://localhost:5601/app/home

On first run, click:

Explore on my own

Useful Kibana pages:

Index Management:
http://localhost:5601/app/management/data/index_management/indices

Dev Tools Console:
http://localhost:5601/app/dev_tools#/console

Stack Monitoring:
http://localhost:5601/app/monitoring

Stack Monitoring may show "No monitoring data found" unless self-monitoring or
Metricbeat is configured. For this project, Index Management and Dev Tools are
the quickest way to monitor progress.

Useful Kibana Dev Tools Commands
Check cluster health:

GET _cluster/health?pretty

List indices:

GET _cat/indices?v

Check the document count:

GET wikipedia_vector_index/_count

View a few small records without dumping long article text:

GET wikipedia_vector_index/_search
{
  "size": 3,
  "_source": ["title", "url", "vector_id"],
  "query": {
    "match_all": {}
  }
}

Search with limited source fields:

GET wikipedia_vector_index/_search
{
  "size": 3,
  "_source": ["title", "url", "text"],
  "query": {
    "match": {
      "title": "Atlantic Ocean"
    }
  }
}

Monitor From The Terminal
Check service status:

docker compose ps

Check app logs:

docker compose logs app

Follow app logs:

docker compose logs -f app

Check Elasticsearch health directly:

docker compose exec -T elasticsearch curl -s \
  'http://localhost:9200/_cluster/health?pretty'

Check document count directly:

docker compose exec -T elasticsearch curl -s \
  'http://localhost:9200/wikipedia_vector_index/_count?pretty'

Investigate From Inside The App Container
Open a shell in the app runtime:

docker compose run app bash

Useful checks inside that shell:

printenv | sort | grep -E '^(ELASTIC|OPENAI|PYTHON)'
python --version
python -m pip show elasticsearch
getent hosts elasticsearch

Python connectivity check:

python - <<'PY'
import os
from elasticsearch import Elasticsearch

client = Elasticsearch([{
    "host": os.environ["ELASTIC_HOST"],
    "port": int(os.environ["ELASTIC_PORT"]),
    "scheme": "http",
}])

print("server_version=", client.info()["version"]["number"])
print("ping=", client.ping())
PY

Clean up one-off run containers after investigation if needed:

docker compose rm

Data and Persistence
The Elasticsearch data is stored in a Docker volume named esdata.

The local data directory is mounted into the app container at:

/app/data

This lets the downloaded and extracted CSV survive app container recreation.

The Docker image intentionally does not include .env, .git, or data because
.dockerignore excludes them.

Clean Up
Stop and remove containers while keeping the Elasticsearch volume:

docker compose down

Stop and remove containers and the Elasticsearch data volume:

docker compose down -v

Use docker compose down -v only when you want to delete the indexed data and
start from a fresh Elasticsearch volume.

Troubleshooting
Could not connect to Elasticsearch
Check that the app image has elasticsearch-py 8.x:

docker compose run --rm app python -m pip show elasticsearch

Expected:

Version: 8.x

Check that the app can resolve and ping the service:

docker compose run --rm app python -c "import os; print(os.environ['ELASTIC_HOST'])"

The .env file should contain:

ELASTIC_HOST=elasticsearch
ELASTIC_PORT=9200

Old logs still show connection errors
docker compose logs app can show logs from an old stopped app container. Recreate
the app service after code or dependency changes:

docker compose up -d --build --force-recreate app

Long search result output
Wikipedia article text can be long. SEARCH_RESULT_PREVIEW_CHARS limits the log
preview. OPENAI_CONTEXT_CHARS limits the text sent to OpenAI.

OpenAI usage cost
The script sends only the top search hit to OpenAI, capped by
OPENAI_CONTEXT_CHARS. Lower that value in .env if you want to reduce token usage.

File Structure
.dockerignore: Excludes .env, .git, data, and generated files from the app image.
.env.sample: Sample environment variables.
Dockerfile: Builds the Python app image.
docker-compose.yml: Defines Elasticsearch, Kibana, and app services.
requirements.txt: Python dependencies.
your_script.py: Main RAG ingestion, search, and answer script.
data/: Local mounted data directory.
readme.txt: Project usage and testing documentation.

Code Pointers
Change the sample question in your_script.py:

question = 'Is the Atlantic the biggest ocean in the world?'

The Elasticsearch index mapping is created in create_index().

The bulk indexing flow is in index_data().

The search result printing logic is in pretty_response().

The OpenAI context cap is applied before ChatCompletion.create().

References
Elasticsearch documentation:
https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html

Kibana documentation:
https://www.elastic.co/guide/en/kibana/current/index.html

OpenAI API documentation:
https://platform.openai.com/docs/api-reference

OpenAI Cookbook Example: Retrieval Augmented Generation with Elasticsearch and OpenAI

License
This project is licensed under the MIT License - see the LICENSE file for details.
