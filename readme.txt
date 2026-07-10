Retrieval Augmented Generation with Elasticsearch and OpenAI
===========================================================

The human-friendly documentation is in README.md.

Quick start:

1. Copy the environment file:

   cp .env.sample .env

2. Edit .env and set:

   OPENAI_API_KEY=sk-your-openai-api-key

3. Build and start the stack:

   docker compose up -d --build

4. Follow the app logs:

   docker compose logs -f app

5. Open Kibana:

   http://localhost:5601/app/home

Useful Kibana pages:

   Index Management:
   http://localhost:5601/app/management/data/index_management/indices

   Dev Tools:
   http://localhost:5601/app/dev_tools#/console

Useful Dev Tools commands:

   GET _cluster/health?pretty
   GET _cat/indices?v
   GET wikipedia_vector_index/_count

Notes:

- The app is a one-shot script. It may exit after indexing/search/answer
  generation finishes.
- OPENAI_CONTEXT_CHARS limits how much retrieved text is sent to OpenAI.
- SEARCH_RESULT_PREVIEW_CHARS limits how much text is printed in logs.
- Full setup, testing, troubleshooting, and code notes are in README.md.
