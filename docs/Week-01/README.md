# Week 01 — LLM Fundamentals

Five days of foundational sessions covering how large language models work, how to call them, and how to build your first retrieval and evaluation systems. By the end of Week 01 you'll have the conceptual grounding and practical API skills to move into Week 02 without gaps.

---

## What You'll Be Able to Do After This Week

- Explain how transformers, attention, and tokenization work at the level needed to make system design decisions
- Write effective zero-shot, few-shot, and chain-of-thought prompts for a range of tasks
- Call the OpenAI and Anthropic APIs — completions, vision, tool use — and handle cost and rate limits correctly
- Build a working semantic search engine using embeddings and cosine similarity
- Implement a complete RAG pipeline from PDF ingestion to grounded answers
- Store and query embeddings in ChromaDB, Pinecone, and Qdrant
- Measure RAG quality using RAGAS and identify hallucination failure modes
- Defend your application against prompt injection, jailbreaks, and content policy violations
- Load and run open-source models via Hugging Face and Ollama

---

## Daily Breakdown

### Day 01 — How LLMs Work + Prompt Engineering

=== "Part 1 — How LLMs Work"

    **The foundation.** Everything in this course — model selection, context window design, chunking strategy, temperature tuning — traces back to understanding what the model is actually doing.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Transformers and Attention](Day-01-Part-1-How-LLMs-Work/01-transformers-and-attention.md) | Self-attention, multi-head attention, why transformers parallelise where RNNs couldn't |
    | [Tokenization](Day-01-Part-1-How-LLMs-Work/02-tokenization.md) | BPE, token counting, why `gpt-4o` and `claude-sonnet-4-5` tokenize differently |
    | [Context Windows](Day-01-Part-1-How-LLMs-Work/03-context-windows.md) | How much the model can "see", KV cache, practical limits |
    | [How LLMs Generate Text](Day-01-Part-1-How-LLMs-Work/04-how-llms-generate-text.md) | Temperature, top-p, top-k, greedy vs sampling, why randomness matters |
    | [Practice Exercises](Day-01-Part-1-How-LLMs-Work/05-practice-exercises.md) | Tokenize text, observe sampling behaviour, compare context window limits |
    | [Interview Questions](Day-01-Part-1-How-LLMs-Work/06-interview-questions.md) | 6 questions you'll face in LLM engineering interviews |

=== "Part 2 — Prompt Engineering"

    **The highest-leverage skill.** A well-crafted prompt turns a mediocre model into an excellent one. Before reaching for fine-tuning or RAG, exhaust what good prompting can do.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Zero-Shot and Few-Shot](Day-01-Part-2-Prompt-Engineering/01-zero-shot-and-few-shot.md) | When each works, how many examples is enough, example selection strategy |
    | [Chain-of-Thought](Day-01-Part-2-Prompt-Engineering/02-chain-of-thought.md) | "Let's think step by step", when CoT helps vs hurts, zero-shot CoT |
    | [Structured Output](Day-01-Part-2-Prompt-Engineering/03-structured-output.md) | JSON mode, schema enforcement, prompt-based vs API-native approaches |
    | [System Prompts](Day-01-Part-2-Prompt-Engineering/04-system-prompts.md) | Persona, scope, format — the three dimensions of a system prompt |
    | [Advanced Prompt Patterns](Day-01-Part-2-Prompt-Engineering/05-prompt-patterns.md) | ReAct, self-consistency, meta-prompting, prompt chaining |
    | [Practice Exercises](Day-01-Part-2-Prompt-Engineering/06-practice-exercises.md) | Rewrite weak prompts, implement CoT, extract structured data |
    | [Interview Questions](Day-01-Part-2-Prompt-Engineering/07-interview-questions.md) | 6 prompting Q&As with model answers |

---

### Day 02 — LLM APIs + Embeddings

=== "Part 1 — OpenAI & Anthropic APIs"

    **The tools you'll use every day.** Two frontier providers with different design philosophies — learn both well enough to build production systems with either and switch between them without rewriting code.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [OpenAI Chat Completions](Day-02-Part-1-OpenAI-and-Anthropic-APIs/01-openai-chat-completions.md) | Messages API, streaming, async client, key parameters |
    | [Vision and Multimodal](Day-02-Part-1-OpenAI-and-Anthropic-APIs/02-vision-and-multimodal.md) | Image inputs, PDF processing, structured extraction from documents |
    | [Tool Use with OpenAI](Day-02-Part-1-OpenAI-and-Anthropic-APIs/03-tool-use-openai.md) | Defining tools, parsing `tool_calls`, parallel tool execution |
    | [Anthropic Messages API](Day-02-Part-1-OpenAI-and-Anthropic-APIs/04-anthropic-messages-api.md) | Key differences from OpenAI, `tool_use` blocks, streaming context manager |
    | [Cost and Rate Limits](Day-02-Part-1-OpenAI-and-Anthropic-APIs/05-cost-and-rate-limits.md) | Token pricing, exponential backoff, budget guardrails |
    | [Practice Exercises](Day-02-Part-1-OpenAI-and-Anthropic-APIs/06-practice-exercises.md) | Build a unified client that works across both APIs |
    | [Interview Questions](Day-02-Part-1-OpenAI-and-Anthropic-APIs/07-interview-questions.md) | API behaviour, error handling, cost estimation |

=== "Part 2 — Embeddings & Semantic Search"

    **Text as geometry.** Once language lives in vector space, you search it with math instead of keywords — the foundation of every RAG system you'll build.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [What Are Embeddings](Day-02-Part-2-Embeddings-and-Semantic-Search/01-what-are-embeddings.md) | Dense vectors, semantic meaning, why similar text clusters together |
    | [Embedding Models](Day-02-Part-2-Embeddings-and-Semantic-Search/02-embedding-models.md) | `text-embedding-3-small`, Sentence Transformers, MTEB benchmark |
    | [Cosine Similarity](Day-02-Part-2-Embeddings-and-Semantic-Search/03-cosine-similarity.md) | Dot product, L2 norm, distance vs similarity, implementation |
    | [Vector Search](Day-02-Part-2-Embeddings-and-Semantic-Search/04-vector-search.md) | Brute force, FAISS, approximate nearest neighbour tradeoffs |
    | [Semantic Search Pipeline](Day-02-Part-2-Embeddings-and-Semantic-Search/05-semantic-search-pipeline.md) | End-to-end: chunk → embed → index → retrieve → rerank |
    | [Practice Exercises](Day-02-Part-2-Embeddings-and-Semantic-Search/06-practice-exercises.md) | Build a semantic FAQ search engine from scratch |
    | [Interview Questions](Day-02-Part-2-Embeddings-and-Semantic-Search/07-interview-questions.md) | Embeddings, similarity, model selection |

---

### Day 03 — RAG Basics + Vector Databases

=== "Part 1 — RAG Basics"

    **The most important system pattern you'll learn.** Retrieval-Augmented Generation solves the core production failure of LLMs: hallucination about facts outside their training data.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [What Is RAG](Day-03-Part-1-RAG-Basics/01-what-is-rag.md) | Architecture, the retrieve-augment-generate loop, when it helps |
    | [Chunking Strategies](Day-03-Part-1-RAG-Basics/02-chunking-strategies.md) | Fixed, semantic, recursive, hierarchical — tradeoffs for each |
    | [Retrieval and Augmentation](Day-03-Part-1-RAG-Basics/03-retrieval-and-augmentation.md) | Context assembly, prompt design, handling retrieval failures |
    | [RAG Pipeline End-to-End](Day-03-Part-1-RAG-Basics/04-rag-pipeline-end-to-end.md) | PDF ingestion → embedding → retrieval → grounded answer |
    | [RAG vs Fine-Tuning](Day-03-Part-1-RAG-Basics/05-rag-vs-fine-tuning.md) | Decision framework — what each solves, when to combine them |
    | [Practice Exercises](Day-03-Part-1-RAG-Basics/06-practice-exercises.md) | Build a minimal RAG system over a document set |
    | [Interview Questions](Day-03-Part-1-RAG-Basics/07-interview-questions.md) | RAG design, failure modes, chunking choices |

=== "Part 2 — Vector Databases"

    **FAISS is for laptops. Production RAG needs persistence, filtering, and CRUD.** Learn the three databases you'll encounter in real stacks.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Vector DB Overview](Day-03-Part-2-Vector-Databases/01-vector-db-overview.md) | Capabilities, managed vs self-hosted, pricing comparison |
    | [ChromaDB](Day-03-Part-2-Vector-Databases/02-chromadb.md) | PersistentClient, collections, metadata filtering operators |
    | [Pinecone](Day-03-Part-2-Vector-Databases/03-pinecone.md) | Serverless indexes, namespaces, sparse-dense hybrid |
    | [Qdrant](Day-03-Part-2-Vector-Databases/04-qdrant.md) | Self-hosted with Docker, payload filtering, on-disk indexing |
    | [Indexing and Filtering](Day-03-Part-2-Vector-Databases/05-indexing-and-filtering.md) | HNSW parameters, metadata filter operators, performance tuning |
    | [Hybrid Search](Day-03-Part-2-Vector-Databases/06-hybrid-search.md) | BM25 + dense, Reciprocal Rank Fusion, when sparse retrieval wins |
    | [Practice Exercises](Day-03-Part-2-Vector-Databases/07-practice-exercises.md) | Port a FAISS index to ChromaDB, add metadata filtering |
    | [Interview Questions](Day-03-Part-2-Vector-Databases/08-interview-questions.md) | DB selection, indexing tradeoffs, filtering gotchas |

---

### Day 04 — Evaluation + Responsible AI

=== "Part 1 — LLM Evaluation"

    **Shipping without evaluation is shipping blind.** You need to measure faithfulness, relevance, and hallucination rate — and verify that your prompt changes actually improved something.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Evaluation Overview](Day-04-Part-1-LLM-Evaluation/01-evaluation-overview.md) | Eval types, metrics vs vibes, designing a test set |
    | [RAGAS Framework](Day-04-Part-1-LLM-Evaluation/02-ragas-framework.md) | Faithfulness, answer relevancy, context recall, context precision |
    | [Hallucination and Faithfulness](Day-04-Part-1-LLM-Evaluation/03-hallucination-and-faithfulness.md) | Why models confabulate, faithfulness scoring, detection approaches |
    | [Relevance Metrics](Day-04-Part-1-LLM-Evaluation/04-relevance-metrics.md) | Answer relevancy, context relevancy, BERTScore |
    | [Human Evaluations](Day-04-Part-1-LLM-Evaluation/05-human-evals.md) | Rubric design, inter-rater reliability, calibration |
    | [Practice Exercises](Day-04-Part-1-LLM-Evaluation/06-practice-exercises.md) | Run RAGAS on your Day 03 RAG pipeline |
    | [Interview Questions](Day-04-Part-1-LLM-Evaluation/07-interview-questions.md) | Eval strategy, metric choice, LLM-as-judge tradeoffs |

=== "Part 2 — Responsible AI & Safety"

    **Deploying LLMs without safety is like deploying a web server without auth.** Your model will face adversarial inputs the moment it's public — build the skills to detect, prevent, and respond.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Jailbreaks and Prompt Injection](Day-04-Part-2-Responsible-AI-and-Safety/01-jailbreaks-and-prompt-injection.md) | Attack taxonomy, indirect injection, defence strategies |
    | [Guardrails](Day-04-Part-2-Responsible-AI-and-Safety/02-guardrails.md) | Input validation, output checking, multi-stage filtering |
    | [Content Filtering](Day-04-Part-2-Responsible-AI-and-Safety/03-content-filtering.md) | OpenAI Moderation API, Presidio PII detection, custom classifiers |
    | [Bias and Fairness](Day-04-Part-2-Responsible-AI-and-Safety/04-bias-and-fairness.md) | Demographic auditing, counterfactual testing, mitigation approaches |
    | [Practice Exercises](Day-04-Part-2-Responsible-AI-and-Safety/05-practice-exercises.md) | Add a guardrail layer to the RAG pipeline from Day 03 |
    | [Interview Questions](Day-04-Part-2-Responsible-AI-and-Safety/06-interview-questions.md) | Safety architecture, prompt injection mitigations |

---

### Day 05 — Hugging Face + Local LLMs

=== "Part 1 — Hugging Face Ecosystem"

    **Where the open-source ML world lives.** 500,000+ models, 200,000+ datasets, and hosted inference — all behind a consistent Python SDK.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Hugging Face Hub](Day-05-Part-1-Hugging-Face-Ecosystem/01-hugging-face-hub.md) | Searching models, model cards, gated models, downloading weights |
    | [Transformers Library](Day-05-Part-1-Hugging-Face-Ecosystem/02-transformers-library.md) | `pipeline()`, AutoTokenizer, AutoModelForCausalLM, inference loop |
    | [Inference API](Day-05-Part-1-Hugging-Face-Ecosystem/03-inference-api.md) | Serverless hosted inference without owning GPU hardware |
    | [Spaces](Day-05-Part-1-Hugging-Face-Ecosystem/04-spaces.md) | Deploying a Gradio demo, forking Spaces, sharing your work |
    | [Datasets and Model Cards](Day-05-Part-1-Hugging-Face-Ecosystem/05-datasets-and-models.md) | Loading datasets, understanding model cards, MTEB leaderboard |
    | [Practice Exercises](Day-05-Part-1-Hugging-Face-Ecosystem/06-practice-exercises.md) | Load a fine-tuned model from Hub and run classification |
    | [Interview Questions](Day-05-Part-1-Hugging-Face-Ecosystem/07-interview-questions.md) | Hub navigation, transformers library patterns |

=== "Part 2 — Local LLMs"

    **No API costs. No data leaving your machine. No rate limits.** Learn to run models locally — and more importantly, when it's actually worth it.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Why Run Locally](Day-05-Part-2-Local-LLMs/01-why-run-locally.md) | Privacy, latency, cost — the three reasons to go local |
    | [Ollama](Day-05-Part-2-Local-LLMs/02-ollama.md) | Install, pull models, OpenAI-compatible REST API, model switching |
    | [llama.cpp](Day-05-Part-2-Local-LLMs/03-llama-cpp.md) | GGUF format, CPU-only inference, Python bindings |
    | [Quantization](Day-05-Part-2-Local-LLMs/04-quantization.md) | Q4, Q8, NF4 — the quality vs memory tradeoff in numbers |
    | [When to Run Locally](Day-05-Part-2-Local-LLMs/05-when-to-run-locally.md) | Decision framework: task, data sensitivity, hardware, cost |
    | [Practice Exercises](Day-05-Part-2-Local-LLMs/06-practice-exercises.md) | Run Llama 3 with Ollama, benchmark against `gpt-4o-mini` |
    | [Interview Questions](Day-05-Part-2-Local-LLMs/07-interview-questions.md) | Local vs API, quantization tradeoffs, hardware requirements |

---

## Prerequisites

- Python 3.10+ with `pip`
- An OpenAI API key (`OPENAI_API_KEY`) — covers most sessions
- An Anthropic API key (`ANTHROPIC_API_KEY`) — used in Day 02 Part 1
- A Hugging Face account and token (`HF_TOKEN`) — used in Day 05

```bash
pip install openai anthropic chromadb sentence-transformers \
            transformers datasets huggingface_hub ragas \
            pinecone-client qdrant-client pypdf2 tiktoken
```

> [!warning] Never hardcode API keys
> Always load keys from environment variables: `os.getenv("OPENAI_API_KEY")`. Every code example in this course follows this pattern.

---

## Week 01 Assignment

Five tasks that build on each session:

1. Make an authenticated API call to both OpenAI and Anthropic and print the response
2. Build a semantic similarity tool: embed two sets of sentences, compute cosine similarity, rank results
3. Build a minimal RAG pipeline over a PDF using ChromaDB and `text-embedding-3-small`
4. Run RAGAS faithfulness and answer relevancy on your RAG pipeline with at least 10 test cases
5. Use function calling to extract structured invoice data from three unstructured text samples

Full brief: [Week 01 Assignment](../Assignments/week-01-assignment.md)

---

> [!success] What Week 01 builds toward
> The skills from this week are the substrate for everything in Week 02. LangChain is just structured API calls. LangGraph is just state machines over prompt chains. Fine-tuning is just gradient descent on the patterns you've already seen. Understand the foundation and Week 02 clicks into place.

**Next:** [Week 02 — Building LLM Applications](../Week-02/README.md)
