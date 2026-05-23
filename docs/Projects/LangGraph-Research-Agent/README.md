# Project 5 — LangGraph Research Agent

Build a multi-node LangGraph agent that takes a research question, plans sub-questions, researches each one, synthesizes findings, and runs a critic loop to ensure quality. This is the most architecturally complex project — it demonstrates stateful, conditional, multi-agent reasoning.

## What you'll build

A LangGraph graph with:
- **Planner node** — decomposes the research question into 3–5 sub-questions
- **Researcher node** — answers each sub-question (in parallel, with asyncio.gather)
- **Writer node** — synthesizes findings into a structured report
- **Critic node** — evaluates quality and provides feedback
- **Conditional router** — loops back to writer if quality is below threshold (max 3 loops)
- A FastAPI endpoint that exposes the agent as a streaming API

## Skills covered

| Skill | Where |
|-------|-------|
| LangGraph StateGraph | [[02-implementation]] |
| Conditional routing | [[02-implementation]] |
| Parallel node execution | [[02-implementation]] |
| MemorySaver checkpointing | [[03-advanced-features]] |
| Agent quality evaluation | [[04-evaluation]] |
| LangSmith tracing | [[03-advanced-features]] |

## Prerequisites

- Week 02 Day 03 Part 1 — AI Agents
- Week 02 Day 03 Part 2 — LangGraph

## Tech stack

```
langchain==0.3.7
langchain-openai==0.2.3
langgraph==0.2.28
langsmith==0.1.147
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
python-dotenv==1.0.1
```

---

[[01-setup]]
