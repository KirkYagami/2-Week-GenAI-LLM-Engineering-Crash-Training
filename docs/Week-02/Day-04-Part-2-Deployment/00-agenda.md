# Agenda — Deployment

**Session length:** 3 hours | **Difficulty:** Intermediate | **Prerequisites:** [[Week-02/Day-04-Part-1-LLMOps/05-observability|Observability]], FastAPI basics, async Python

---

## What you will build today

A production-ready FastAPI service that wraps an LLM pipeline with streaming responses, async handling, semantic caching, and a serverless-compatible structure.

---

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00–0:20 | FastAPI wrappers: request/response models, auth | [[01-fastapi-wrappers]] |
| 0:20–0:55 | Streaming responses: SSE and chunked transfer | [[02-streaming-responses]] |
| 0:55–1:20 | Async patterns: concurrent requests, background tasks | [[03-async-patterns]] |
| 1:20–1:45 | Serverless deployment: AWS Lambda, Modal, Fly.io | [[04-serverless]] |
| 1:45–2:10 | Caching strategies: exact match and semantic cache | [[05-caching-strategies]] |
| 2:10–2:45 | Practice exercises | [[06-practice-exercises]] |
| 2:45–3:00 | Interview questions review | [[07-interview-questions]] |

---

## Setup

```bash
pip install fastapi uvicorn sse-starlette pydantic openai httpx
```

---

[[Week-02/Day-04-Part-1-LLMOps/07-interview-questions|← LLMOps]] | [[01-fastapi-wrappers|FastAPI →]]
