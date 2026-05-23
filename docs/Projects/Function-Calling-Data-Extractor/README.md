# Project 4 — Function-Calling Data Extractor

Build a structured data extraction API that uses OpenAI's tool_use to pull fields from unstructured text — emails, invoices, contracts, support tickets — and validate the result against a Pydantic schema.

## What you'll build

A FastAPI service that:
- Accepts raw text or a PDF upload
- Uses function calling to extract structured fields (entity, date, amount, sentiment, category)
- Validates the extraction with Pydantic — invalid fields are flagged, not silently dropped
- Supports multiple extraction schemas (invoice, support ticket, job description)
- Returns a confidence score based on field completeness

## Skills covered

| Skill | Where |
|-------|-------|
| OpenAI tools API (4-step cycle) | [[02-implementation]] |
| Pydantic schema generation | [[02-implementation]] |
| Multi-schema extraction | [[03-advanced-features]] |
| Extraction quality evaluation | [[04-evaluation]] |
| F1 vs ground truth | [[04-evaluation]] |

## Prerequisites

- Week 02 Day 02 Part 2 — Function Calling and Tool Use
- Week 02 Day 04 Part 2 — Deployment

## Tech stack

```
openai==1.51.0
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
pymupdf==1.24.11
python-dotenv==1.0.1
httpx==0.27.2
```

---

[[01-setup]]
