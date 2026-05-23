# Deployment — RAG Q&A Chatbot

## Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-chatbot
```

Note: the `-v` flag mounts your local `chroma_db/` into the container so the index persists across container restarts.

## Fly.io deployment

```toml
# fly.toml
app = 'rag-chatbot'
primary_region = 'iad'

[build]
  dockerfile = 'Dockerfile'

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[env]
  PORT = "8000"
  COLLECTION_NAME = "documents"

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 1

[[mounts]]
  source = "chroma_data"
  destination = "/app/chroma_db"
```

```bash
# Deploy
flyctl launch --name rag-chatbot
flyctl secrets set OPENAI_API_KEY=sk-...
flyctl volumes create chroma_data --size 1
flyctl deploy

# Verify
flyctl status
curl https://rag-chatbot.fly.dev/health
```

> [!info] ChromaDB needs persistent storage on Fly.io
> The `[[mounts]]` section creates a persistent volume at `/app/chroma_db`. Without this, the vector index is lost on every deploy. Run `python ingest.py` once after the first deploy (or build ingest into a startup script).

## Running ingestion on deployment

```bash
# SSH into the running instance to run ingestion
flyctl ssh console
cd /app && python ingest.py
exit
```

Or add to a `Makefile`:

```makefile
deploy:
	flyctl deploy
	flyctl ssh console --command "python /app/ingest.py"
```

## Environment variables summary

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `CHROMA_PERSIST_PATH` | No | `./chroma_db` | Path for ChromaDB storage |
| `COLLECTION_NAME` | No | `documents` | ChromaDB collection name |

---

[[04-evaluation]] | [[06-interview-questions]]
