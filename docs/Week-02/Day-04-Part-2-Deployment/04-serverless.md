# Serverless Deployment

Serverless platforms handle scaling, infrastructure, and cold start management so you focus on your application code. For LLM services that see bursty, unpredictable traffic, serverless is often the right default — you pay per request rather than for idle capacity.

## Learning objectives

- Structure a FastAPI app for serverless deployment (Modal, AWS Lambda)
- Handle cold starts in LLM services
- Deploy to Modal with GPU support for local models
- Package dependencies correctly for Lambda
- Know the tradeoffs between serverless and always-on deployment

---

## Serverless considerations for LLM apps

```python
# Cold start problem:
# Serverless functions spin up from zero on first request.
# LLM-related cold starts:
#   - Loading a local model (llama.cpp, transformers): 10-60 seconds
#   - Importing heavy libraries (torch, transformers): 5-10 seconds
#   - Initializing OpenAI client: <100ms (negligible)

# Solutions:
#   1. Keep-warm pings (scheduled requests to prevent cold starts)
#   2. Minimum instance count = 1 (always-on, costs money)
#   3. Lazy loading (initialize on first actual request, not import)
#   4. Light dependencies (openai + fastapi cold-start in ~1-2s)
```

---

## Modal deployment

Modal is the easiest serverless platform for Python LLM apps. It handles containers, GPUs, and scaling automatically.

```python
# modal_app.py
import modal
import os

app = modal.App("llm-service")

# Define the container image
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "openai", "fastapi", "uvicorn", "pydantic"
)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=60,
    # For GPU-accelerated local models:
    # gpu="T4",
    # min_containers=1,  # Keep warm
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from openai import OpenAI
    from pydantic import BaseModel

    api = FastAPI()
    client = OpenAI()  # API key from Modal secret

    class ChatRequest(BaseModel):
        message: str

    @api.post("/chat")
    async def chat(request: ChatRequest):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": request.message}],
        )
        return {"response": response.choices[0].message.content}

    @api.get("/health")
    async def health():
        return {"status": "ok"}

    return api

# Deploy with: modal deploy modal_app.py
# Test with:   modal run modal_app.py
```

---

## AWS Lambda with Mangum

```python
# lambda_app.py
import os
from fastapi import FastAPI
from mangum import Mangum
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Set in Lambda env vars

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request.message}],
    )
    return {"response": response.choices[0].message.content}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Lambda handler — wraps FastAPI with ASGI adapter
handler = Mangum(app, lifespan="off")

# Deployment:
# 1. pip install mangum -t ./package
# 2. cp lambda_app.py ./package/
# 3. cd package && zip -r ../lambda.zip .
# 4. aws lambda create-function --function-name llm-service \
#       --runtime python3.11 --handler lambda_app.handler \
#       --zip-file fileb://lambda.zip --role arn:aws:iam::...

# requirements.txt for Lambda layer:
# openai
# fastapi
# mangum
# pydantic
```

---

## Fly.io deployment (always-on, simple)

```
# fly.toml
app = 'my-llm-service'
primary_region = 'iad'

[build]
  dockerfile = 'Dockerfile'

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true    # Scale to zero when idle
  auto_start_machines = true   # Auto-start on traffic
  min_machines_running = 0     # True serverless (0 = scale to zero)

[env]
  PORT = "8000"

[[vm]]
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1
```

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```
# Deploy with:
flyctl launch
flyctl secrets set OPENAI_API_KEY=sk-...
flyctl deploy
```

---

## Serverless tradeoff table

| Platform | Cold start | GPU support | Cost model | Best for |
|----------|-----------|-------------|------------|----------|
| Modal | 1–5s (API), 30–90s (GPU) | Yes | Per compute-second | GPU models, ML experiments |
| AWS Lambda | 1–3s | No | Per request | High-volume API wrappers |
| Fly.io | 2–5s | No | Per machine-second | Always-on, simple setup |
| Google Cloud Run | 2–4s | No | Per request | GCP ecosystem |
| Railway | 0s (always-on) | No | Per month (flat) | Dev, staging environments |

> [!warning] Lambda has a 15-minute execution timeout and 10GB memory limit
> Streaming responses don't work on standard Lambda (response streaming requires Lambda response streaming, which is a separate feature). If you need streaming, use Modal, Fly.io, or Cloud Run instead.

> [!tip] Use Modal for local model serving — it's the easiest GPU serverless option
> Modal's GPU cold start (loading a 7B model) takes 30–90 seconds. Set `min_containers=1` to keep one instance warm, and `max_containers=10` to auto-scale under load. The cost of one warm container (~$0.30/hr for T4) is usually worth eliminating cold starts.

---

[[03-async-patterns]] | [[05-caching-strategies]]
