# Project 2 — AI Writing Assistant

Build a multi-step writing pipeline that takes a topic or rough draft and produces polished content with tone control, style enforcement, and streaming output. This project focuses on chaining LLM calls, managing intermediate state, and building a usable API around a complex prompt workflow.

## What you'll build

A FastAPI service that:
- Accepts a topic or draft text and a target style (professional, casual, technical, persuasive)
- Runs a multi-step pipeline: outline → draft → refine → style-check
- Streams the final output token-by-token
- Supports tone adjustment without regenerating from scratch
- Returns structured metadata: word count, reading level, key points extracted

## Skills covered

| Skill | Where |
|-------|-------|
| Multi-step prompt chains | [[02-implementation]] |
| LCEL pipelines | [[02-implementation]] |
| Streaming with intermediate steps | [[02-implementation]] |
| Style transfer prompts | [[03-advanced-features]] |
| Output quality evaluation | [[04-evaluation]] |
| FastAPI with async chains | [[05-deployment]] |

## Prerequisites

- Week 02 Day 01 Part 1 — LangChain Fundamentals
- Week 02 Day 04 Part 2 — Deployment (streaming)

## Tech stack

```
openai==1.51.0
langchain==0.3.7
langchain-openai==0.2.3
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
python-dotenv==1.0.1
```

## Result

A working writing assistant API with streaming, tone control, and a structured output format that you can demo with any topic in 10 seconds.

---

[[01-setup]]
