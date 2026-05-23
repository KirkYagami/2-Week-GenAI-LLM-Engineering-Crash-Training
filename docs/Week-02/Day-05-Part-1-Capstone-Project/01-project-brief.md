# Project Brief — Capstone

The capstone is an open-ended LLM application of your choice, evaluated on depth of integration, code quality, and measurable outcomes — not novelty. A well-built FAQ bot beats a half-built research agent every time.

## Learning objectives

- Scope an LLM project to something completable in a half-day
- Make deliberate architectural choices and document the reasoning
- Build production-quality code: error handling, async, secrets via env vars
- Quantitatively evaluate your system, not just demo it

---

## Project options

Pick one of these or propose your own (confirm scope with the instructor first).

### Option A — Customer Support Bot with RAG

A FastAPI service that answers customer questions by retrieving from a product documentation corpus, with streaming responses and a cache layer.

**Minimum viable scope:**
- Ingest a set of markdown or PDF docs into ChromaDB
- Retrieve top-5 chunks for each user question
- Generate a grounded response with citations
- Stream the response via SSE
- Track cache hit rate and average latency over a test set of 20 questions

**Stretch:**
- Add semantic caching to handle rephrased questions
- Deploy to Modal or Fly.io
- Evaluate faithfulness with RAGAS

---

### Option B — Research Agent with LangGraph

A multi-step agent that takes a research question, searches the web (or a document corpus), synthesizes findings, and produces a structured report.

**Minimum viable scope:**
- LangGraph graph with at least: planner → researcher → writer → critic nodes
- Conditional routing: critic rejects low-quality drafts (loop back to writer)
- Tool calls for search or document retrieval
- Final structured output (JSON or markdown report)

**Stretch:**
- Persistent memory across sessions with MemorySaver
- LangSmith tracing for all nodes
- Cost tracking per run

---

### Option C — Document Intelligence Pipeline

An API that accepts a PDF upload, extracts structured data using function calling, evaluates extraction quality, and returns validated JSON.

**Minimum viable scope:**
- Accept a PDF, extract text (PyMuPDF or pdfplumber)
- Use OpenAI tool_use to extract a Pydantic-validated schema (company, date, amounts, entities)
- Validate the extraction — flag low-confidence fields
- Return structured JSON with a confidence score per field

**Stretch:**
- Fine-tune or prompt-engineer for a specific document type (invoices, contracts)
- Compare extraction quality against a ground-truth test set
- Build a FastAPI endpoint with streaming progress updates

---

### Option D — Bring Your Own Project

Your own LLM application idea. Must include at minimum:
- At least 4 components from the required list in [[00-agenda]]
- A quantitative evaluation (not just a demo)
- A working FastAPI endpoint

---

## Scope management

The most common capstone failure mode is overscoping. Two hours of real implementation time is not enough to build a novel LLM framework. Use this filter:

1. Can I write the core data flow in 30 lines of Python? (If not, scope down.)
2. Do I already know how to implement every component? (If not, pick a simpler component you do know.)
3. Can I define a measurable success criterion right now? (If not, the project isn't scoped enough.)

> [!tip] Start with the hardest integration first
> The riskiest part of any LLM project is usually the data pipeline (ingestion, chunking, retrieval quality) or the LLM integration itself (prompt engineering, output parsing). Build that first. If it doesn't work, you want to know in the first 30 minutes, not the last 10.

---

[[00-agenda]] | [[02-architecture-design]]
