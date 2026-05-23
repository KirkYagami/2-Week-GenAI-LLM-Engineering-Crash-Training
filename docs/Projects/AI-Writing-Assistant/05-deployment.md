# Deployment — AI Writing Assistant

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

Note: `--workers 2` runs two uvicorn worker processes. With an async FastAPI app, a single worker handles concurrent requests well — add workers if CPU becomes the bottleneck (e.g., for the reading level analysis which is CPU-bound).

## Railway deployment (simple, no persistent storage needed)

The writing assistant has no persistent state — every request is independent. This makes it easy to deploy to Railway or any PaaS.

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Initialize and deploy
railway init
railway up

# Set environment variables
railway variables set OPENAI_API_KEY=sk-...
railway variables set DEFAULT_MODEL=gpt-4o-mini
```

Railway auto-detects the `Dockerfile` and deploys. No `fly.toml` needed.

## Fly.io deployment

```toml
# fly.toml
app = 'ai-writing-assistant'
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
flyctl launch --name ai-writing-assistant
flyctl secrets set OPENAI_API_KEY=sk-...
flyctl deploy
```

## Rate limiting

The writing pipeline makes 4 sequential LLM calls per request. To prevent abuse:

```python
from collections import defaultdict, deque
import time

_request_windows: dict[str, deque] = defaultdict(deque)

def check_rate_limit(client_ip: str, max_per_minute: int = 5) -> None:
    now = time.time()
    window = _request_windows[client_ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= max_per_minute:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=429,
            detail="Rate limit: 5 writing requests per minute",
            headers={"Retry-After": "60"},
        )
    window.append(now)
```

Add to the `/write` endpoint:

```python
from fastapi import Request

@app.post("/write")
async def write(request: WriteRequest, http_request: Request):
    client_ip = http_request.client.host
    check_rate_limit(client_ip)
    # ... rest of handler
```

---

[[04-evaluation]] | [[06-interview-questions]]
