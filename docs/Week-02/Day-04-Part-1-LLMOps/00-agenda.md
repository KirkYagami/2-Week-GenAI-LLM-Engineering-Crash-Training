# Agenda — LLMOps

**Session length:** 3 hours | **Difficulty:** Intermediate | **Prerequisites:** [[Week-02/Day-03-Part-2-LangGraph/05-multi-agent-orchestration|Multi-Agent Orchestration]], basic Python logging

---

## What you will build today

A production-ready LLM request pipeline with structured logging, cost tracking, LangSmith tracing, and latency benchmarks.

---

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00–0:20 | Tracing and structured logging for LLM calls | [[01-tracing-and-logging]] |
| 0:20–0:50 | LangSmith: traces, datasets, evaluations | [[02-langsmith]] |
| 0:50–1:15 | Cost tracking: token counting, budget alerts | [[03-cost-tracking]] |
| 1:15–1:45 | Latency optimization: caching, batching, model selection | [[04-latency-optimization]] |
| 1:45–2:05 | Observability: metrics, dashboards, alerting | [[05-observability]] |
| 2:05–2:45 | Practice exercises | [[06-practice-exercises]] |
| 2:45–3:00 | Interview questions review | [[07-interview-questions]] |

---

## Setup

```bash
pip install langsmith openai tiktoken prometheus-client
export LANGCHAIN_API_KEY="your-langsmith-key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="my-llm-project"
```

---

[[Week-02/Day-03-Part-2-LangGraph/07-interview-questions|← LangGraph]] | [[01-tracing-and-logging|Tracing →]]
