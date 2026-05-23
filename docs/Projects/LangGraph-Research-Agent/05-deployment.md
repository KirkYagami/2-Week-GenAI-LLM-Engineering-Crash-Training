# Deployment — LangGraph Research Agent

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Timeout considerations

A full research run (4 nodes, up to 3 writer loops) can take 20–60 seconds. Configure timeouts accordingly:

```toml
# fly.toml
[http_service]
  internal_port = 8000

[checks]
  [checks.health]
    grace_period = "10s"
    interval = "30s"
    method = "GET"
    path = "/health"
    timeout = "5s"
    type = "http"
```

Set client timeout to at least 90 seconds:

```python
# In your client code
with httpx.Client(timeout=90.0) as client:
    result = client.post(url, json={"question": q})
```

## Async task queue for long-running requests

For production, avoid holding HTTP connections open for 60s. Use a task queue:

```python
import uuid
from fastapi import BackgroundTasks

_results: dict[str, dict] = {}

@app.post("/research/async")
async def research_async(request: ResearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    _results[task_id] = {"status": "running"}

    async def run_and_store():
        result = await run(request.question)
        _results[task_id] = {"status": "done", **result}

    background_tasks.add_task(run_and_store)
    return {"task_id": task_id, "status_url": f"/research/status/{task_id}"}

@app.get("/research/status/{task_id}")
async def research_status(task_id: str):
    result = _results.get(task_id)
    if not result:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Task not found")
    return result
```

The client polls `/research/status/{task_id}` until `status == "done"`.

---

[[04-evaluation]] | [[06-interview-questions]]
