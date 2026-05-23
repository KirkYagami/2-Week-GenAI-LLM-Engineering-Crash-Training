# GenAI & LLM Engineering — 2-Week Crash Training

Ten sessions. Build production-ready LLM systems. Walk into your next interview with working projects and real evaluation numbers.

---

## What You'll Be Able to Do

After completing both weeks, you will be able to:

- Explain how attention, tokenization, and context windows work — and why they constrain system design
- Build and evaluate RAG pipelines: chunking, retrieval, reranking, and RAGAS metrics
- Write production FastAPI endpoints with async patterns, Server-Sent Events streaming, and caching
- Design and implement LangGraph agents with conditional routing and human-in-the-loop checkpoints
- Fine-tune a model with QLoRA, merge the adapter weights, and serve it on CPU
- Design LLM systems under constraints: latency, cost, accuracy, and privacy tradeoffs

---

## Course Structure

=== "Week 01 — LLM Fundamentals"

    | Day | Part 1 | Part 2 |
    |-----|--------|--------|
    | 01 | [How LLMs Work](Week-01/Day-01-Part-1-How-LLMs-Work/00-agenda.md) | [Prompt Engineering](Week-01/Day-01-Part-2-Prompt-Engineering/00-agenda.md) |
    | 02 | [OpenAI & Anthropic APIs](Week-01/Day-02-Part-1-OpenAI-and-Anthropic-APIs/00-agenda.md) | [Embeddings & Semantic Search](Week-01/Day-02-Part-2-Embeddings-and-Semantic-Search/00-agenda.md) |
    | 03 | [RAG Basics](Week-01/Day-03-Part-1-RAG-Basics/00-agenda.md) | [Vector Databases](Week-01/Day-03-Part-2-Vector-Databases/00-agenda.md) |
    | 04 | [LLM Evaluation](Week-01/Day-04-Part-1-LLM-Evaluation/00-agenda.md) | [Responsible AI & Safety](Week-01/Day-04-Part-2-Responsible-AI-and-Safety/00-agenda.md) |
    | 05 | [Hugging Face Ecosystem](Week-01/Day-05-Part-1-Hugging-Face-Ecosystem/00-agenda.md) | [Local LLMs](Week-01/Day-05-Part-2-Local-LLMs/00-agenda.md) |

=== "Week 02 — Building LLM Applications"

    | Day | Part 1 | Part 2 |
    |-----|--------|--------|
    | 01 | [LangChain Fundamentals](Week-02/Day-01-Part-1-LangChain-Fundamentals/00-agenda.md) | [Advanced RAG](Week-02/Day-01-Part-2-Advanced-RAG/00-agenda.md) |
    | 02 | [Fine-Tuning](Week-02/Day-02-Part-1-Fine-Tuning/00-agenda.md) | [Function Calling & Tool Use](Week-02/Day-02-Part-2-Function-Calling-and-Tool-Use/00-agenda.md) |
    | 03 | [AI Agents](Week-02/Day-03-Part-1-AI-Agents/00-agenda.md) | [LangGraph](Week-02/Day-03-Part-2-LangGraph/00-agenda.md) |
    | 04 | [LLMOps](Week-02/Day-04-Part-1-LLMOps/00-agenda.md) | [Deployment](Week-02/Day-04-Part-2-Deployment/00-agenda.md) |
    | 05 | [Capstone Project](Week-02/Day-05-Part-1-Capstone-Project/00-agenda.md) | [Mock Interview & Portfolio](Week-02/Day-05-Part-2-Mock-Interview-and-Portfolio/00-agenda.md) |

---

## 6 Guided Projects

Each project ships with a setup guide, full implementation, advanced features, evaluation scripts, deployment instructions, and interview Q&A.

| # | Project | What You Build | Core Skills |
|---|---------|----------------|-------------|
| 1 | [RAG Q&A Chatbot](Projects/RAG-QA-Chatbot/README.md) | PDF ingestion → embedding → retrieval → streaming answers with citations | ChromaDB, CrossEncoder reranking, RAGAS |
| 2 | [AI Writing Assistant](Projects/AI-Writing-Assistant/README.md) | 4-stage LCEL pipeline: outline → draft → refine → style check | LangChain, async streaming, SSE |
| 3 | [Document Summarizer](Projects/Document-Summarizer-with-Eval/README.md) | Map-reduce summarization with faithfulness and coherence evaluation | asyncio.gather, LLM-as-judge |
| 4 | [Function-Calling Extractor](Projects/Function-Calling-Data-Extractor/README.md) | Extract typed, validated data structures from unstructured text | OpenAI tools, Pydantic, schema generation |
| 5 | [LangGraph Research Agent](Projects/LangGraph-Research-Agent/README.md) | Planner → researcher → writer → critic loop with quality gating | LangGraph StateGraph, conditional routing |
| 6 | [Fine-Tuned Classifier](Projects/Fine-Tuned-Classifier/README.md) | QLoRA fine-tune a small model, merge weights, compare vs prompt-only | LoRA, PEFT, SFTTrainer, CPU serving |

---

## Quick Start

```bash
git clone https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training.git
cd 2-Week-GenAI-LLM-Engineering-Crash-Training
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY and ANTHROPIC_API_KEY
mkdocs serve           # open http://127.0.0.1:8000
```

> [!info] What API keys do you need?
> An **OpenAI API key** covers most of the course. An **Anthropic key** is used in Week 01 Day 02. **LangSmith** is optional but recommended for Week 02 tracing exercises — it's free to sign up.

---

## Reference Materials

**Cheat Sheets** — one-page code references for every major tool

[OpenAI API](Cheat-Sheets/openai-api-cheat-sheet.md) · [Anthropic API](Cheat-Sheets/anthropic-api-cheat-sheet.md) · [LangChain LCEL](Cheat-Sheets/langchain-cheat-sheet.md) · [LangGraph](Cheat-Sheets/langgraph-cheat-sheet.md) · [ChromaDB](Cheat-Sheets/chromadb-cheat-sheet.md) · [Prompt Engineering](Cheat-Sheets/prompt-engineering-cheat-sheet.md) · [RAG](Cheat-Sheets/rag-cheat-sheet.md) · [Fine-Tuning](Cheat-Sheets/fine-tuning-cheat-sheet.md) · [AI Agents](Cheat-Sheets/agents-cheat-sheet.md) · [LLMOps](Cheat-Sheets/llmops-cheat-sheet.md)

**Interview Preparation**

[Overview & Study Plan](Interview-Preparation/README.md) · [Mock Interview Simulator](Interview-Preparation/08-mock-interview-simulator.md)

**Resources**

[Essential Papers](Resources/papers.md) · [Courses](Resources/courses.md) · [YouTube Channels](Resources/youtube-channels.md) · [Practice Platforms](Resources/practice-platforms.md)

**Other**

[Glossary](glossary.md) · [Progress Tracker](progress-tracker.md) · [Companion Notebooks](notebooks.md)

---

> [!success] The point of this course
> LLM engineering is a systems discipline. The goal is not to know the most papers — it is to build systems that work reliably, evaluate them honestly, and explain your decisions clearly. Every note, project, and exercise here is in service of that.
