# Companion Notebooks

Interactive Jupyter notebooks that accompany the course notes. Each notebook covers one session's key concepts with runnable code — no setup required when opened in Google Colab.

> [!tip] Running in Google Colab
> Click the "Open in Colab" link next to any notebook. You'll need to add your API keys as Colab Secrets (the key icon in the left sidebar) — `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Never paste keys directly into notebook cells.

> [!info] Running locally
> Notebooks live in the `notebooks/` directory at the repo root. Run `pip install -r requirements.txt` and `jupyter lab` to get started.

---

## Week 01 Notebooks

| Notebook | Topics Covered | Open in Colab |
|----------|---------------|---------------|
| `week01-day01-llm-fundamentals.ipynb` | Tokenization, context windows, temperature sampling | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day01-llm-fundamentals.ipynb) |
| `week01-day01-prompt-engineering.ipynb` | Zero-shot, few-shot, CoT, structured output | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day01-prompt-engineering.ipynb) |
| `week01-day02-apis.ipynb` | OpenAI and Anthropic API calls, streaming, tool use | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day02-apis.ipynb) |
| `week01-day02-embeddings.ipynb` | Embedding models, cosine similarity, semantic search | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day02-embeddings.ipynb) |
| `week01-day03-rag-basics.ipynb` | Chunking, ChromaDB ingestion, end-to-end RAG pipeline | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day03-rag-basics.ipynb) |
| `week01-day03-vector-databases.ipynb` | ChromaDB, Pinecone, Qdrant, hybrid search | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day03-vector-databases.ipynb) |
| `week01-day04-evaluation.ipynb` | RAGAS faithfulness, answer relevancy, LLM-as-judge | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day04-evaluation.ipynb) |
| `week01-day04-responsible-ai.ipynb` | Prompt injection detection, OpenAI Moderation API, PII | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day04-responsible-ai.ipynb) |
| `week01-day05-huggingface.ipynb` | Hub navigation, `pipeline()`, Inference API | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day05-huggingface.ipynb) |
| `week01-day05-local-llms.ipynb` | Ollama API calls, quantization comparison | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week01-day05-local-llms.ipynb) |

---

## Week 02 Notebooks

| Notebook | Topics Covered | Open in Colab |
|----------|---------------|---------------|
| `week02-day01-langchain.ipynb` | LCEL pipelines, memory, async, streaming | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day01-langchain.ipynb) |
| `week02-day01-advanced-rag.ipynb` | CrossEncoder reranking, HyDE, multi-query, compression | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day01-advanced-rag.ipynb) |
| `week02-day02-fine-tuning.ipynb` | QLoRA setup, SFTTrainer, evaluation, merge and save | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day02-fine-tuning.ipynb) |
| `week02-day02-function-calling.ipynb` | OpenAI tools cycle, Anthropic tool_use, Pydantic extraction | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day02-function-calling.ipynb) |
| `week02-day03-agents.ipynb` | ReAct from scratch, tool registration, memory strategies | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day03-agents.ipynb) |
| `week02-day03-langgraph.ipynb` | StateGraph, reducers, conditional routing, MemorySaver | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day03-langgraph.ipynb) |
| `week02-day04-llmops.ipynb` | LangSmith tracing, cost tracking, latency benchmarking | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day04-llmops.ipynb) |
| `week02-day04-deployment.ipynb` | FastAPI + AsyncOpenAI, SSE streaming, semantic cache | [Open](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/week02-day04-deployment.ipynb) |

---

## Project Starter Notebooks

Each project has a starter notebook with data loading and client setup done. Exercise cells are left blank for you to complete.

| Project | Starter Notebook |
|---------|-----------------|
| RAG Q&A Chatbot | [starter.ipynb](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/project-rag-chatbot-starter.ipynb) |
| AI Writing Assistant | [starter.ipynb](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/project-writing-assistant-starter.ipynb) |
| Document Summarizer | [starter.ipynb](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/project-summarizer-starter.ipynb) |
| Function-Calling Extractor | [starter.ipynb](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/project-extractor-starter.ipynb) |
| LangGraph Research Agent | [starter.ipynb](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/project-research-agent-starter.ipynb) |
| Fine-Tuned Classifier | [starter.ipynb](https://github.com/KirkYagami/2-Week-GenAI-LLM-Engineering-Crash-Training/blob/main/notebooks/project-fine-tuned-classifier-starter.ipynb) |

---

> [!warning] Notebooks vs course notes
> Notebooks show working code in isolation. The course notes explain the *why*. Use notebooks to run and experiment, but read the notes to understand the design decisions. Don't skip the notes because the notebook runs — that's the difference between copying and learning.
