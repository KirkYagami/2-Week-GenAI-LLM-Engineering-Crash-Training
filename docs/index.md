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
    | 01 | [[Day-01-Part-1-How-LLMs-Work\|How LLMs Work]] | [[Day-01-Part-2-Prompt-Engineering\|Prompt Engineering]] |
    | 02 | [[Day-02-Part-1-OpenAI-and-Anthropic-APIs\|OpenAI & Anthropic APIs]] | [[Day-02-Part-2-Embeddings-and-Semantic-Search\|Embeddings & Semantic Search]] |
    | 03 | [[Day-03-Part-1-RAG-Basics\|RAG Basics]] | [[Day-03-Part-2-Vector-Databases\|Vector Databases]] |
    | 04 | [[Day-04-Part-1-LLM-Evaluation\|LLM Evaluation]] | [[Day-04-Part-2-Responsible-AI-and-Safety\|Responsible AI & Safety]] |
    | 05 | [[Day-05-Part-1-Hugging-Face-Ecosystem\|Hugging Face Ecosystem]] | [[Day-05-Part-2-Local-LLMs\|Local LLMs]] |

=== "Week 02 — Building LLM Applications"

    | Day | Part 1 | Part 2 |
    |-----|--------|--------|
    | 01 | [[Day-01-Part-1-LangChain-Fundamentals\|LangChain Fundamentals]] | [[Day-01-Part-2-Advanced-RAG\|Advanced RAG]] |
    | 02 | [[Day-02-Part-1-Fine-Tuning\|Fine-Tuning]] | [[Day-02-Part-2-Function-Calling-and-Tool-Use\|Function Calling & Tool Use]] |
    | 03 | [[Day-03-Part-1-AI-Agents\|AI Agents]] | [[Day-03-Part-2-LangGraph\|LangGraph]] |
    | 04 | [[Day-04-Part-1-LLMOps\|LLMOps]] | [[Day-04-Part-2-Deployment\|Deployment]] |
    | 05 | [[Day-05-Part-1-Capstone-Project\|Capstone Project]] | [[Day-05-Part-2-Mock-Interview-and-Portfolio\|Mock Interview & Portfolio]] |

---

## 6 Guided Projects

Each project ships with a setup guide, full implementation, advanced features, evaluation scripts, deployment instructions, and interview Q&A.

| # | Project | What You Build | Core Skills |
|---|---------|----------------|-------------|
| 1 | [[RAG-QA-Chatbot\|RAG Q&A Chatbot]] | PDF ingestion → embedding → retrieval → streaming answers with citations | ChromaDB, CrossEncoder reranking, RAGAS |
| 2 | [[AI-Writing-Assistant\|AI Writing Assistant]] | 4-stage LCEL pipeline: outline → draft → refine → style check | LangChain, async streaming, SSE |
| 3 | [[Document-Summarizer-with-Eval\|Document Summarizer]] | Map-reduce summarization with faithfulness and coherence evaluation | asyncio.gather, LLM-as-judge |
| 4 | [[Function-Calling-Data-Extractor\|Function-Calling Extractor]] | Extract typed, validated data structures from unstructured text | OpenAI tools, Pydantic, schema generation |
| 5 | [[LangGraph-Research-Agent\|LangGraph Research Agent]] | Planner → researcher → writer → critic loop with quality gating | LangGraph StateGraph, conditional routing |
| 6 | [[Fine-Tuned-Classifier\|Fine-Tuned Classifier]] | QLoRA fine-tune a small model, merge weights, compare vs prompt-only | LoRA, PEFT, SFTTrainer, CPU serving |

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

[[openai-api-cheat-sheet|OpenAI API]] · [[anthropic-api-cheat-sheet|Anthropic API]] · [[langchain-cheat-sheet|LangChain LCEL]] · [[langgraph-cheat-sheet|LangGraph]] · [[chromadb-cheat-sheet|ChromaDB]] · [[prompt-engineering-cheat-sheet|Prompt Engineering]] · [[rag-cheat-sheet|RAG]] · [[fine-tuning-cheat-sheet|Fine-Tuning]] · [[agents-cheat-sheet|AI Agents]] · [[llmops-cheat-sheet|LLMOps]]

**Interview Preparation**

[[Interview-Preparation/README|Overview & Study Plan]] · [[08-mock-interview-simulator|Mock Interview Simulator]]

**Resources**

[[papers|Essential Papers]] · [[courses|Courses]] · [[youtube-channels|YouTube Channels]] · [[practice-platforms|Practice Platforms]]

**Other**

[[glossary|Glossary]] · [[progress-tracker|Progress Tracker]] · [[Notebooks|Companion Notebooks]]

---

> [!success] The point of this course
> LLM engineering is a systems discipline. The goal is not to know the most papers — it is to build systems that work reliably, evaluate them honestly, and explain your decisions clearly. Every note, project, and exercise here is in service of that.
