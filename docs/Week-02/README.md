# Week 02 — Building LLM Applications

Five days of applied sessions turning Week 01 fundamentals into production systems. You build real applications, deploy them with FastAPI, evaluate them with real metrics, and finish with a capstone project ready to demo in an interview.

> [!info] Prerequisites
> Week 02 assumes you completed Week 01 or have equivalent experience: you can call the OpenAI API, you understand what embeddings are, and you've built at least one RAG pipeline.

---

## What You'll Be Able to Do After This Week

- Build composable LLM pipelines with LangChain LCEL using async and streaming patterns
- Implement advanced RAG: reranking with CrossEncoder, HyDE, multi-query retrieval
- Fine-tune a model with QLoRA using PEFT and SFTTrainer, then serve it via FastAPI
- Extract structured, validated data from unstructured text using function calling and Pydantic
- Design and implement LangGraph agents with conditional routing, reducers, and checkpointing
- Add observability to LLM systems: LangSmith tracing, cost tracking, latency benchmarking
- Deploy production FastAPI endpoints with SSE streaming, semantic caching, and serverless hosting
- Present a working, evaluated LLM system and explain every architecture decision

---

## Daily Breakdown

### Day 01 — LangChain + Advanced RAG

=== "Part 1 — LangChain Fundamentals"

    **The most widely used LLM application framework.** LangChain solves two problems: composability (chaining LLM calls, tools, and data sources) and portability (switching providers without rewriting logic).

    | Topic | What You'll Learn |
    |-------|------------------|
    | [LangChain Overview](Day-01-Part-1-LangChain-Fundamentals/01-langchain-overview.md) | Why it exists, core abstractions, LCEL vs legacy chains |
    | [Chains and Prompts](Day-01-Part-1-LangChain-Fundamentals/02-chains-and-prompts.md) | `ChatPromptTemplate`, few-shot templates, output parsers |
    | [Memory](Day-01-Part-1-LangChain-Fundamentals/03-memory.md) | `RunnableWithMessageHistory`, buffer memory, token-limited memory |
    | [LCEL](Day-01-Part-1-LangChain-Fundamentals/04-lcel.md) | Pipe operator, `.ainvoke()`, `.astream()`, `RunnablePassthrough`, `RunnableParallel` |
    | [Practice Exercises](Day-01-Part-1-LangChain-Fundamentals/05-practice-exercises.md) | Build a multi-step content pipeline with LCEL |
    | [Interview Questions](Day-01-Part-1-LangChain-Fundamentals/06-interview-questions.md) | LCEL internals, memory backends, async vs sync |

=== "Part 2 — Advanced RAG"

    **Basic RAG gets you 70% quality.** The remaining 30% comes from addressing specific failure modes: irrelevant chunks, query-document embedding mismatch, and context noise.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Reranking](Day-01-Part-2-Advanced-RAG/01-reranking.md) | CrossEncoder scoring, Cohere Rerank API, bi-encoder vs cross-encoder tradeoff |
    | [HyDE](Day-01-Part-2-Advanced-RAG/02-hyde.md) | Hypothetical Document Embeddings — bridge the query-document gap |
    | [Multi-Query Retrieval](Day-01-Part-2-Advanced-RAG/03-multi-query.md) | Generate multiple query variants, union results, improve recall |
    | [Contextual Compression](Day-01-Part-2-Advanced-RAG/04-contextual-compression.md) | Strip irrelevant text from retrieved chunks before generation |
    | [Advanced RAG Patterns](Day-01-Part-2-Advanced-RAG/05-advanced-rag-patterns.md) | Combining techniques: HyDE + multi-query + rerank in one pipeline |
    | [Practice Exercises](Day-01-Part-2-Advanced-RAG/06-practice-exercises.md) | Add reranking to your Week 01 RAG pipeline, measure Recall@3 improvement |
    | [Interview Questions](Day-01-Part-2-Advanced-RAG/07-interview-questions.md) | When to use HyDE, reranking costs, failure mode diagnosis |

---

### Day 02 — Fine-Tuning + Function Calling

=== "Part 1 — Fine-Tuning"

    **When prompting and RAG aren't enough.** Fine-tuning teaches the model a new style, format, or domain — but it's expensive to do wrong. This session gives you a clear decision framework and a working QLoRA pipeline.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Fine-Tuning Overview](Day-02-Part-1-Fine-Tuning/01-fine-tuning-overview.md) | Full fine-tuning vs PEFT vs RAG vs prompting — when each applies |
    | [LoRA and QLoRA](Day-02-Part-1-Fine-Tuning/02-lora-and-qlora.md) | Low-rank decomposition, rank `r`, alpha, NF4 quantization, memory math |
    | [PEFT](Day-02-Part-1-Fine-Tuning/03-peft.md) | `get_peft_model()`, trainable parameter count, `merge_and_unload()` |
    | [Training Data Preparation](Day-02-Part-1-Fine-Tuning/04-training-data.md) | Chat template format, dataset quality over quantity, deduplication |
    | [When to Fine-Tune](Day-02-Part-1-Fine-Tuning/05-when-to-fine-tune.md) | Decision framework: task type, data availability, latency budget |
    | [Practice Exercises](Day-02-Part-1-Fine-Tuning/06-practice-exercises.md) | Fine-tune a classifier with QLoRA, compare accuracy vs prompt-only |
    | [Interview Questions](Day-02-Part-1-Fine-Tuning/07-interview-questions.md) | LoRA parameters, training instability, catastrophic forgetting |

=== "Part 2 — Function Calling & Tool Use"

    **Structured output, reliably.** Function calling lets you extract validated data structures from unstructured text — no regex, no brittle string parsing.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Function Calling Overview](Day-02-Part-2-Function-Calling-and-Tool-Use/01-function-calling-overview.md) | The 4-step cycle: define → call → execute → return |
    | [OpenAI Tools API](Day-02-Part-2-Function-Calling-and-Tool-Use/02-openai-tools.md) | JSON schema definitions, `tool_calls` parsing, `tool_choice` |
    | [Anthropic Tool Use](Day-02-Part-2-Function-Calling-and-Tool-Use/03-anthropic-tool-use.md) | `tool_use` content blocks, `tool_result` messages, key API differences |
    | [Structured Extraction](Day-02-Part-2-Function-Calling-and-Tool-Use/04-structured-extraction.md) | Pydantic models, `model_json_schema()`, `.with_structured_output()` |
    | [Parallel Tool Calls](Day-02-Part-2-Function-Calling-and-Tool-Use/05-parallel-tool-calls.md) | Multiple tools per response, execution ordering, result assembly |
    | [Practice Exercises](Day-02-Part-2-Function-Calling-and-Tool-Use/06-practice-exercises.md) | Extract invoice and support ticket data from raw text |
    | [Interview Questions](Day-02-Part-2-Function-Calling-and-Tool-Use/07-interview-questions.md) | Reliability vs JSON mode, parallel calls, schema design |

---

### Day 03 — AI Agents + LangGraph

=== "Part 1 — AI Agents"

    **Chains execute a fixed path. Agents decide the path at runtime.** Learn when agents are the right tool and how to build them without the magic — just a loop, a prompt, and tools.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [What Are AI Agents](Day-03-Part-1-AI-Agents/01-what-are-agents.md) | Agents vs chains, the perception-action loop, when complexity is warranted |
    | [The ReAct Loop](Day-03-Part-1-AI-Agents/02-react-loop.md) | Thought → Action → Observation, stop sequences, loop termination |
    | [Planning Strategies](Day-03-Part-1-AI-Agents/03-planning-strategies.md) | Sequential, parallel, tree-of-thought — tradeoffs and failure modes |
    | [Tool-Calling Agents](Day-03-Part-1-AI-Agents/04-tool-calling-agents.md) | Tool registration, execution, error recovery, result formatting |
    | [Memory Strategies](Day-03-Part-1-AI-Agents/05-memory-strategies.md) | In-context, sliding window, summarized, vector-backed memory |
    | [Practice Exercises](Day-03-Part-1-AI-Agents/06-practice-exercises.md) | Build a research agent with web search + calculator tools |
    | [Interview Questions](Day-03-Part-1-AI-Agents/07-interview-questions.md) | Agent reliability, tool design, failure recovery, token cost |

=== "Part 2 — LangGraph"

    **Stateful, controllable agent graphs.** LangGraph gives you explicit state management, conditional routing, and human-in-the-loop — things a raw agent loop can't provide reliably.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [LangGraph Overview](Day-03-Part-2-LangGraph/01-langgraph-overview.md) | Why LangGraph exists, StateGraph vs simple chains, key abstractions |
    | [Nodes and Edges](Day-03-Part-2-LangGraph/02-nodes-and-edges.md) | Node functions, `START`/`END`, `add_edge`, `add_conditional_edges` |
    | [Stateful Graphs](Day-03-Part-2-LangGraph/03-stateful-graphs.md) | `TypedDict` state, `Annotated` reducers, `operator.add` for accumulation |
    | [Conditional Routing](Day-03-Part-2-LangGraph/04-conditional-routing.md) | Router functions, quality gates, retry loops, cycle detection |
    | [Multi-Agent Orchestration](Day-03-Part-2-LangGraph/05-multi-agent-orchestration.md) | Supervisor patterns, parallel subgraphs, `MemorySaver`, `interrupt_before` |
    | [Practice Exercises](Day-03-Part-2-LangGraph/06-practice-exercises.md) | Build a writer-critic loop that reruns until quality score ≥ 0.80 |
    | [Interview Questions](Day-03-Part-2-LangGraph/07-interview-questions.md) | State reducers, checkpointing, human-in-the-loop design |

---

### Day 04 — LLMOps + Deployment

=== "Part 1 — LLMOps"

    **You can't improve what you can't measure.** Add observability to your LLM systems — tracing every request, tracking every dollar, catching regressions before users do.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [Tracing and Logging](Day-04-Part-1-LLMOps/01-tracing-and-logging.md) | Structured logging, span tracing, what to capture per request |
    | [LangSmith](Day-04-Part-1-LLMOps/02-langsmith.md) | `@traceable`, dataset creation, `evaluate()`, regression detection |
    | [Cost Tracking](Day-04-Part-1-LLMOps/03-cost-tracking.md) | Token counting with tiktoken, pricing dict, per-request cost logging |
    | [Latency Optimization](Day-04-Part-1-LLMOps/04-latency-optimization.md) | Parallel calls, exact-match cache, model selection, streaming perception |
    | [Observability](Day-04-Part-1-LLMOps/05-observability.md) | Metrics to track, alerting thresholds, dashboard design |
    | [Practice Exercises](Day-04-Part-1-LLMOps/06-practice-exercises.md) | Instrument your RAG pipeline with LangSmith and cost tracking |
    | [Interview Questions](Day-04-Part-1-LLMOps/07-interview-questions.md) | Monitoring strategy, cost control, regression detection |

=== "Part 2 — Deployment"

    **A model that isn't deployed isn't a product.** Wrap your pipelines in production FastAPI services with streaming, async, caching, and serverless hosting.

    | Topic | What You'll Learn |
    |-------|------------------|
    | [FastAPI Wrappers](Day-04-Part-2-Deployment/01-fastapi-wrappers.md) | `lifespan`, Pydantic request/response models, `APIKeyHeader` auth |
    | [Streaming Responses](Day-04-Part-2-Deployment/02-streaming-responses.md) | `StreamingResponse`, SSE format (`data: {}\n\n`), `X-Accel-Buffering: no` |
    | [Async Patterns](Day-04-Part-2-Deployment/03-async-patterns.md) | `AsyncOpenAI`, `asyncio.Semaphore`, `asyncio.gather`, `BackgroundTasks` |
    | [Serverless Deployment](Day-04-Part-2-Deployment/04-serverless.md) | Modal (`@app.function`), AWS Lambda + Mangum, Fly.io cold start tuning |
    | [Caching Strategies](Day-04-Part-2-Deployment/05-caching-strategies.md) | SHA256 exact-match cache, semantic cache with cosine similarity threshold |
    | [Practice Exercises](Day-04-Part-2-Deployment/06-practice-exercises.md) | Add streaming + semantic cache to your RAG endpoint |
    | [Interview Questions](Day-04-Part-2-Deployment/07-interview-questions.md) | SSE vs WebSocket, async client design, cache invalidation |

---

### Day 05 — Capstone + Mock Interview

=== "Part 1 — Capstone Project"

    **Everything comes together.** Design, build, evaluate, and present a production LLM application integrating at least four course components.

    | Resource | Purpose |
    |----------|---------|
    | [Project Brief](Day-05-Part-1-Capstone-Project/01-project-brief.md) | Three project options with full scope definitions |
    | [Architecture Design](Day-05-Part-1-Capstone-Project/02-architecture-design.md) | Design doc template, Mermaid diagrams, component selection guide |
    | [Implementation](Day-05-Part-1-Capstone-Project/03-implementation.md) | Complete working code for all three options |
    | [Evaluation and Testing](Day-05-Part-1-Capstone-Project/04-evaluation-and-testing.md) | Eval scripts per project option, API endpoint testing |
    | [Presentation Guide](Day-05-Part-1-Capstone-Project/05-presentation-guide.md) | 5-minute demo structure, what interviewers listen for |
    | [Submission Checklist](Day-05-Part-1-Capstone-Project/06-submission-checklist.md) | Self-assessment rubric, required vs stretch deliverables |

    **Three project options:**

    - **Option A — Customer Support Bot with RAG** — ChromaDB retrieval, streaming FastAPI, RAGAS evaluation
    - **Option B — Research Agent with LangGraph** — planner + researcher + writer + critic with quality gating
    - **Option C — Document Intelligence Pipeline** — multi-format ingestion, structured extraction, LLM-as-judge

=== "Part 2 — Mock Interview & Portfolio"

    **Convert two weeks of learning into something usable in a job search.** Resume polish, GitHub portfolio, rehearsed answers, and a full mock interview with feedback.

    | Resource | Purpose |
    |----------|---------|
    | [Resume Checklist](Day-05-Part-2-Mock-Interview-and-Portfolio/01-resume-checklist.md) | Skills section format, bullet formula, common mistakes |
    | [Portfolio and GitHub](Day-05-Part-2-Mock-Interview-and-Portfolio/02-portfolio-and-github.md) | Repo structure, README template, what interviewers actually look at |
    | [Technical Interview Questions](Day-05-Part-2-Mock-Interview-and-Portfolio/03-technical-interview-questions.md) | 12 Q&As across RAG, APIs, agents, deployment, fine-tuning, evaluation |
    | [System Design Practice](Day-05-Part-2-Mock-Interview-and-Portfolio/04-system-design-practice.md) | 3 full problems with clarifying questions, reference architectures, tradeoffs |
    | [Mock Interview Script](Day-05-Part-2-Mock-Interview-and-Portfolio/05-mock-interview-script.md) | Complete 45-minute mock with 10 questions, interviewer notes, debrief template |

---

## Week 02 Tech Stack

```bash
pip install langchain langchain-openai langchain-anthropic langchain-community \
            langchain-cohere langgraph langsmith \
            transformers peft trl bitsandbytes accelerate \
            fastapi uvicorn sse-starlette httpx \
            ragas sentence-transformers chromadb \
            modal mangum tiktoken
```

> [!warning] GPU required for fine-tuning
> Day 02 Part 1 fine-tuning exercises require a CUDA GPU with at least 16 GB VRAM for QLoRA on a 7B model. Use Google Colab (A100 runtime) or Kaggle (T4 GPU) if you don't have local hardware.

---

## Week 02 Assignment

Build one of three capstone options to production quality:

- Working FastAPI endpoint with streaming
- Async patterns used correctly throughout
- At least 4 course components integrated
- Quantitative evaluation with ≥ 10 test cases and results in README

Full brief: [Week 02 Assignment](../Assignments/week-02-assignment.md)

---

> [!success] What this week proves
> A working, evaluated, documented project that you built from scratch is worth more in an interview than any certificate. When you can say "faithfulness score 0.87 on 50 test cases, p95 latency 1.2s, cost $0.003 per request" — that's the conversation that gets offers.

**Previous:** [Week 01 — LLM Fundamentals](../Week-01/README.md)
