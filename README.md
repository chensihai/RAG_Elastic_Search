# Retrieval Augmented Generation with Elasticsearch and OpenAI

This project demonstrates a small Retrieval Augmented Generation (RAG) workflow
using Elasticsearch for vector search and OpenAI for embeddings and answer
generation.

## What It Does

1. Starts Elasticsearch and Kibana with Docker Compose.
2. Downloads a pre-embedded Wikipedia dataset from the OpenAI examples CDN.
3. Creates an Elasticsearch vector index named `wikipedia_vector_index`.
4. Bulk indexes the Wikipedia article vectors.
5. Embeds a sample question with OpenAI.
6. Runs a k-NN search in Elasticsearch.
7. Sends capped retrieved context to OpenAI Chat Completion to generate an answer.

## Services

| Service | Purpose | URL |
| --- | --- | --- |
| Elasticsearch | Stores text, metadata, and dense vectors | <http://localhost:9200> |
| Kibana | Inspect indices, document counts, and search results | <http://localhost:5601> |
| app | Python one-shot ingestion/search/answer script | N/A |

## Important Version Note

Elasticsearch runs as version `8.9.1`. The Python client must stay on 8.x:

```text
elasticsearch>=8.9,<9
```

If pip installs `elasticsearch-py` 9.x, Elasticsearch 8 rejects the client
headers and the app may print:

```text
Could not connect to Elasticsearch
```

## Prerequisites

- Docker and Docker Compose
- An OpenAI API key

## Setup

Clone the repository:

```bash
git clone https://github.com/chensihai/RAG_Elastic_Search
cd RAG_Elastic_Search
```

Create your local environment file:

```bash
cp .env.sample .env
```

Edit `.env` and set your OpenAI key:

```env
OPENAI_API_KEY=sk-your-openai-api-key
```

Optional cost/log controls:

```env
OPENAI_CONTEXT_CHARS=3000
SEARCH_RESULT_PREVIEW_CHARS=500
```

`OPENAI_CONTEXT_CHARS` caps how much retrieved article text is sent to OpenAI.
Lower it to reduce token usage.

`SEARCH_RESULT_PREVIEW_CHARS` caps how much retrieved text is printed in logs.

## Run

Build and start the stack:

```bash
docker compose up -d --build
```

Follow app progress:

```bash
docker compose logs -f app
```

Expected progress:

```text
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
```

The app is a one-shot script. It is normal for the `app` container to exit after
indexing, search, and answer generation finish.

After code changes, recreate only the app service:

```bash
docker compose up -d --build --force-recreate app
```

## Monitor With Kibana

Open Kibana:

<http://localhost:5601/app/home>

On first run, click **Explore on my own**.

Useful pages:

- Index Management: <http://localhost:5601/app/management/data/index_management/indices>
- Dev Tools Console: <http://localhost:5601/app/dev_tools#/console>
- Stack Monitoring: <http://localhost:5601/app/monitoring>

Stack Monitoring may show **No monitoring data found** unless self-monitoring or
Metricbeat is configured. For this project, Index Management and Dev Tools are
the quickest ways to monitor progress.

## Kibana Dev Tools Commands

Check cluster health:

```http
GET _cluster/health?pretty
```

List indices:

```http
GET _cat/indices?v
```

Check document count:

```http
GET wikipedia_vector_index/_count
```

View small records without dumping long article text:

```http
GET wikipedia_vector_index/_search
{
  "size": 3,
  "_source": ["title", "url", "vector_id"],
  "query": {
    "match_all": {}
  }
}
```

Search by title:

```http
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
```

## Monitor From Terminal

Check services:

```bash
docker compose ps
```

Check logs:

```bash
docker compose logs app
docker compose logs -f app
```

Check Elasticsearch directly:

```bash
docker compose exec -T elasticsearch curl -s \
  'http://localhost:9200/_cluster/health?pretty'
```

Check document count directly:

```bash
docker compose exec -T elasticsearch curl -s \
  'http://localhost:9200/wikipedia_vector_index/_count?pretty'
```

## Investigate Inside The App Container

Open a shell in the app runtime:

```bash
docker compose run app bash
```

Useful checks:

```bash
printenv | sort | grep -E '^(ELASTIC|OPENAI|PYTHON)'
python --version
python -m pip show elasticsearch
getent hosts elasticsearch
```

Python connectivity check:

```bash
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
```

Clean up one-off run containers after investigation:

```bash
docker compose rm
```

## Data And Persistence

- Elasticsearch data is stored in the Docker volume `esdata`.
- Local `data/` is mounted into the app container at `/app/data`.
- `.dockerignore` keeps `.env`, `.git`, `data/`, and generated files out of the
  Docker image.

Delete containers but keep indexed data:

```bash
docker compose down
```

Delete containers and indexed data:

```bash
docker compose down -v
```

Use `docker compose down -v` only when you want a fresh Elasticsearch volume.

## Troubleshooting

### Could Not Connect To Elasticsearch

Check that the app image has `elasticsearch-py` 8.x:

```bash
docker compose run --rm app python -m pip show elasticsearch
```

Expected:

```text
Version: 8.x
```

Check the app environment:

```bash
docker compose run --rm app python -c "import os; print(os.environ['ELASTIC_HOST'])"
```

Expected `.env` values:

```env
ELASTIC_HOST=elasticsearch
ELASTIC_PORT=9200
```

### Old Logs Still Show Connection Errors

`docker compose logs app` can show logs from an old stopped app container.
Recreate the app service:

```bash
docker compose up -d --build --force-recreate app
```

### Long Search Result Output

Wikipedia article text can be long.

- `SEARCH_RESULT_PREVIEW_CHARS` limits log preview text.
- `OPENAI_CONTEXT_CHARS` limits text sent to OpenAI.

### OpenAI Usage Cost

The script sends only the top search hit to OpenAI, capped by
`OPENAI_CONTEXT_CHARS`. Lower it in `.env` to reduce token usage.

## File Structure

| Path | Purpose |
| --- | --- |
| `.dockerignore` | Excludes `.env`, `.git`, `data/`, and generated files from the app image |
| `.env.sample` | Sample environment variables |
| `Dockerfile` | Builds the Python app image |
| `docker-compose.yml` | Defines Elasticsearch, Kibana, and app services |
| `requirements.txt` | Python dependencies |
| `your_script.py` | Main RAG ingestion, search, and answer script |
| `data/` | Local mounted data directory |
| `README.md` | Human-friendly project documentation |
| `readme.txt` | Plain-text pointer and quick start |

## Code Pointers

Change the sample question in `your_script.py`:

```python
question = 'Is the Atlantic the biggest ocean in the world?'
```

- `create_index()` creates the Elasticsearch mapping.
- `index_data()` bulk indexes the dataset.
- `pretty_response()` prints search results.
- The OpenAI context cap is applied before `ChatCompletion.create()`.

## References

- [Elasticsearch documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [Kibana documentation](https://www.elastic.co/guide/en/kibana/current/index.html)
- [OpenAI API documentation](https://platform.openai.com/docs/api-reference)
- OpenAI Cookbook Example: Retrieval Augmented Generation with Elasticsearch and OpenAI

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details if one is added to the repository.
