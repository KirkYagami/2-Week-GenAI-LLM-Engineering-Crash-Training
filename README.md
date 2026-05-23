# 2-Week GenAI & LLM Engineering Crash Training

A focused, hands-on bootcamp taking you from transformer fundamentals to production-ready LLM systems in 10 sessions.

**Live site:** https://llm.learnwithnickk.com/ &nbsp;·&nbsp; **Trainer:** [Nikhil Sharma](https://www.linkedin.com/in/-nikhil/)

---

## What This Is

Ten half-day sessions. No fluff. By the end you will have:

- Built **6 production-grade projects** with working FastAPI endpoints and real evaluation metrics
- Understood the internals well enough to make architecture decisions — RAG vs fine-tuning, sync vs async, exact-match vs semantic cache
- Written evaluation scripts that produce numbers, not vibes
- Prepared for technical interviews at companies building with LLMs

**Who it's for:** Engineers and data scientists who want to move from "I've called the OpenAI API" to "I can design, build, and ship LLM systems."

**Prerequisites:** Python fluency, basic familiarity with HTTP APIs. No ML background required.

---

## What You'll Build

| # | Project | Stack |
|---|---------|-------|
| 1 | **RAG Q&A Chatbot** — ingest PDFs, embed, retrieve, answer with citations | ChromaDB · OpenAI · FastAPI |
| 2 | **AI Writing Assistant** — 4-stage LCEL pipeline with streaming output | LangChain · OpenAI · FastAPI |
| 3 | **Document Summarizer with Eval** — map-reduce summarization + RAGAS evaluation | OpenAI · RAGAS · asyncio |
| 4 | **Function-Calling Data Extractor** — structured extraction from unstructured text | OpenAI tools · Pydantic · FastAPI |
| 5 | **LangGraph Research Agent** — planner + researcher + writer + critic with critic loop | LangGraph · OpenAI · asyncio |
| 6 | **Fine-Tuned Classifier** — QLoRA fine-tune a small model, compare vs prompt-only baseline | Transformers · PEFT · FastAPI |

Every project ships with: setup guide · full implementation · advanced features · evaluation scripts · deployment instructions · interview Q&A.

---

## Course Structure

### Week 01 — LLM Fundamentals

| Day | Part 1 | Part 2 |
|-----|--------|--------|
| 01 | How LLMs Work — transformers, tokenization, attention, context windows | Prompt Engineering — zero-shot, few-shot, CoT, structured output, system prompts |
| 02 | OpenAI & Anthropic APIs — completions, vision, tool use, cost, rate limits | Embeddings & Semantic Search — models, cosine similarity, vector search |
| 03 | RAG Basics — chunking, retrieval, augmented generation, RAG vs fine-tuning | Vector Databases — ChromaDB, Pinecone, Qdrant, indexing, hybrid search |
| 04 | LLM Evaluation — RAGAS, hallucination, faithfulness, relevance, human evals | Responsible AI & Safety — jailbreaks, guardrails, content filtering, bias |
| 05 | Hugging Face Ecosystem — Hub, Transformers library, Inference API, Spaces | Local LLMs — Ollama, llama.cpp, quantization, when to run locally |

### Week 02 — Building LLM Applications

| Day | Part 1 | Part 2 |
|-----|--------|--------|
| 01 | LangChain Fundamentals — LCEL, chains, memory, `.ainvoke()` / `.astream()` | Advanced RAG — reranking, HyDE, multi-query, contextual compression |
| 02 | Fine-Tuning — LoRA, QLoRA, PEFT, training data preparation | Function Calling & Tool Use — OpenAI tools, Anthropic tool_use, structured extraction |
| 03 | AI Agents — ReAct loop, planning, tool-calling agents, memory strategies | LangGraph — StateGraph, conditional routing, multi-agent orchestration |
| 04 | LLMOps — tracing, LangSmith, cost tracking, latency optimization | Deployment — FastAPI, streaming (SSE), async patterns, serverless, caching |
| 05 | End-to-End Capstone Project | Mock Interview & Portfolio Review |

---

## Cheat Sheets

Ten reference sheets with minimal working code for every major tool:

| Sheet | What's in it |
|-------|-------------|
| OpenAI API | Chat completions, async, streaming, embeddings, function calling, structured output, error handling |
| Anthropic API | Messages API, streaming context manager, tool use cycle, key differences from OpenAI |
| LangChain LCEL | Pipe operator, structured output, RAG chain, memory with RunnableWithMessageHistory |
| LangGraph | StateGraph minimal pattern, reducers, conditional edges, async nodes, MemorySaver, interrupt_before |
| ChromaDB | PersistentClient, collection CRUD, metadata filter operators, distance-to-similarity conversion |
| Prompt Engineering | Zero-shot/few-shot/CoT patterns, temperature guide, debugging checklist, anti-patterns |
| RAG | Full pipeline, chunking strategies, HyDE/multi-query/hybrid retrieval, CrossEncoder reranking |
| Fine-Tuning | Complete QLoRA setup, LoRA config, SFTTrainer, chat template format, memory requirements |
| AI Agents | ReAct loop, tool registration, in-context/sliding/summary/vector memory strategies |
| LLMOps | LangSmith `@traceable` and `evaluate()`, cost tracking, latency optimization patterns |

---

## Interview Preparation

- **Topic Q&A** across 7 areas: prompt engineering, RAG, LLM APIs, agents/LangGraph, fine-tuning, LLMOps, system design
- **Mock interview simulator** — 10 questions in 4 difficulty rounds with collapsible model answers
- **Resume checklist** with skills section format and the bullet formula that gets callbacks
- **Portfolio and GitHub strategy** — what interviewers actually look at
- **3 full system design problems** — customer support bot, batch processing pipeline, coding assistant — with reference architectures and tradeoffs

---

## Tech Stack

```
openai              anthropic           langchain           langgraph
langsmith           chromadb            sentence-transformers
transformers        peft                fastapi             ragas
```

---

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training.git
cd 2-Week-GenAI-LLM-Engineering-Crash-Training
pip install -r requirements.txt
```

**2. Configure API keys**

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=...       # optional — enables tracing in Week 02
HF_TOKEN=...                # optional — needed for gated Hugging Face models
```

**3. Serve the docs locally**

```bash
mkdocs serve
# open http://127.0.0.1:8000
```

---

## Repo Structure

```
/
├── docs/
│   ├── index.md                   # site homepage
│   ├── Week-01/                   # 5 days × 2 parts × 6–8 notes each
│   ├── Week-02/                   # 5 days × 2 parts × 6–8 notes each
│   ├── Projects/                  # 6 guided projects × 7 files each
│   │   ├── RAG-QA-Chatbot/
│   │   ├── AI-Writing-Assistant/
│   │   ├── Document-Summarizer-with-Eval/
│   │   ├── Function-Calling-Data-Extractor/
│   │   ├── LangGraph-Research-Agent/
│   │   └── Fine-Tuned-Classifier/
│   ├── Cheat-Sheets/              # 10 reference sheets
│   ├── Interview-Preparation/     # Q&A, mock interview simulator, system design
│   ├── Assignments/               # week-01 and week-02 assignments with rubrics
│   ├── Resources/                 # papers, courses, YouTube channels, practice platforms
│   ├── glossary.md                # 60+ key LLM engineering terms
│   └── progress-tracker.md        # student self-tracking checklist
├── notebooks/                     # companion .ipynb files
├── mkdocs.yml
├── requirements.txt
└── .env.example
```

---

## Assignments

**Week 01 Assignment** — 5 tasks:
1. Make an API call to both OpenAI and Anthropic
2. Build a semantic similarity tool using embeddings
3. Build a minimal RAG pipeline with ChromaDB
4. Run RAGAS evaluation on your RAG pipeline
5. Use structured output (function calling) to extract data from text

**Week 02 Assignment** — Capstone project (your choice of three options):
- Customer Support Bot with RAG
- Research Agent with LangGraph
- Document Intelligence Pipeline

Must include: working FastAPI endpoint, async patterns, error handling, quantitative evaluation with results, README with setup and eval numbers.

---

Made with ❤️ by [Nikhil Sharma](https://www.linkedin.com/in/-nikhil/)
