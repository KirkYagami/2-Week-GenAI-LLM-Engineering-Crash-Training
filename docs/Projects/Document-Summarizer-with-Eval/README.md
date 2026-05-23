# Project 3 — Document Summarizer with Eval

Build a hierarchical document summarization pipeline with a complete RAGAS evaluation suite. This project teaches you to evaluate LLM outputs quantitatively — a skill most junior engineers skip but every serious production team requires.

## What you'll build

A FastAPI service that:
- Accepts long documents (PDF, markdown, plain text) up to 50 pages
- Summarizes using a map-reduce strategy (chunk → chunk summaries → final summary)
- Generates different summary formats: bullet points, executive summary, TL;DR
- Evaluates output quality using RAGAS and custom metrics
- Tracks quality scores over time so you can measure improvements

## Skills covered

| Skill | Where |
|-------|-------|
| Map-reduce summarization | [[02-implementation]] |
| Hierarchical chunking | [[01-setup]] |
| RAGAS evaluation pipeline | [[04-evaluation]] |
| Custom evaluation metrics | [[04-evaluation]] |
| LangSmith experiment tracking | [[03-advanced-features]] |

## Prerequisites

- Week 01 Day 01 Part 1 — How LLMs Work (context windows)
- Week 01 Day 04 Part 1 — LLM Evaluation
- Week 02 Day 04 Part 1 — LLMOps

## Tech stack

```
openai==1.51.0
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
pymupdf==1.24.11
ragas==0.2.3
langchain-openai==0.2.3
langsmith==0.1.147
python-dotenv==1.0.1
```

---

[[01-setup]]
