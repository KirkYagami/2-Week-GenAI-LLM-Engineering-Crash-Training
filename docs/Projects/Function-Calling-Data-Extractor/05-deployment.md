# Deployment — Function-Calling Data Extractor

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Fly.io

```toml
# fly.toml
app = 'data-extractor'
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
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1
```

```bash
flyctl launch --name data-extractor
flyctl secrets set OPENAI_API_KEY=sk-...
flyctl deploy
curl https://data-extractor.fly.dev/health
```

## API key authentication

Add simple API key auth to prevent abuse:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
VALID_KEYS = set(os.getenv("API_KEYS", "test-key").split(","))

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

Use as a dependency:

```python
@app.post("/extract/text")
async def extract_from_text(request: TextExtractRequest, _: str = Depends(verify_api_key)):
    ...
```

Set multiple keys with: `flyctl secrets set API_KEYS=key1,key2,key3`

---

[[04-evaluation]] | [[06-interview-questions]]
