# 2-Week GenAI & LLM Engineering Crash Training

A focused, hands-on course taking you from LLM fundamentals to production-ready agents in 10 sessions.

**Live site:** https://llm.learnwithnickk.com/

## What You'll Build

By the end of Week 02 you will have built 6 production-grade projects:

1. **RAG Q&A Chatbot** — ingest PDFs, embed, retrieve, answer with citations
2. **AI Writing Assistant** — multi-step prompt chain with tone/style control and streaming
3. **Document Summarizer with Eval** — hierarchical summarization + RAGAS evaluation suite
4. **Function-Calling Data Extractor** — structured extraction from unstructured text
5. **LangGraph Research Agent** — planner + search + writer + critic with conditional routing
6. **Fine-Tuned Classifier** — LoRA fine-tune a small model, compare vs prompt-only baseline

## Course Structure

### Week 01 — LLM Fundamentals

| Day | Topics |
|-----|--------|
| 01 | How LLMs Work · Prompt Engineering |
| 02 | OpenAI & Anthropic APIs · Embeddings & Semantic Search |
| 03 | RAG Basics · Vector Databases |
| 04 | LLM Evaluation · Responsible AI & Safety |
| 05 | Hugging Face Ecosystem · Local LLMs |

### Week 02 — Building LLM Applications

| Day | Topics |
|-----|--------|
| 01 | LangChain Fundamentals · Advanced RAG |
| 02 | Fine-Tuning · Function Calling & Tool Use |
| 03 | AI Agents · LangGraph |
| 04 | LLMOps · Deployment |
| 05 | End-to-End Capstone · Mock Interview & Portfolio |

## Tech Stack

`openai` · `anthropic` · `langchain` · `langgraph` · `langsmith` · `chromadb` · `sentence-transformers` · `transformers` · `peft` · `fastapi` · `ragas`

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Run the Docs Site Locally

```bash
mkdocs serve
```

---

Made with ❤️ by [Nikhil Sharma](https://www.linkedin.com/in/-nikhil/)
