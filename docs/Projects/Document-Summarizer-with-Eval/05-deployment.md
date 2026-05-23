# Deployment — Document Summarizer

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

> [!info] PyMuPDF requires system libraries
> The `libmupdf-dev` apt package is needed for PyMuPDF (fitz) to compile. If you see `ImportError: libmupdf.so not found`, this is the fix.

## Fly.io

```toml
# fly.toml
app = 'document-summarizer'
primary_region = 'iad'

[build]
  dockerfile = 'Dockerfile'

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 1
```

```bash
flyctl launch --name document-summarizer
flyctl secrets set OPENAI_API_KEY=sk-...
flyctl deploy
```

## File upload size limits

Fly.io and most platforms have request body limits. Set the limit in FastAPI:

```python
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# Increase upload limit to 10MB
from starlette.datastructures import UploadFile as StarletteUploadFile
StarletteUploadFile.spool_max_size = 10 * 1024 * 1024
```

For nginx (if using a reverse proxy), add to the server block:
```
client_max_body_size 10M;
```

## Request timeout tuning

Large documents take longer. Set appropriate timeouts:

```python
# For a 50-page PDF with 10 chunks:
# map step: 10 concurrent calls × ~1s each = ~1-2s
# reduce step: ~1s
# Total: 3-5s

# Set FastAPI response timeout accordingly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "30"]
```

---

[[04-evaluation]] | [[06-interview-questions]]
