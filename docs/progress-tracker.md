# Progress Tracker

Copy this checklist into your own notes and check off each item as you complete it. Reading ≠ done — mark an item complete only when you've run the code and can explain the concept out loud.

---

## Week 01 — LLM Fundamentals

### Day 01 — How LLMs Work + Prompt Engineering

**Part 1 — How LLMs Work**
- [ ] Read: Transformers and Attention
- [ ] Can explain: what self-attention is and why it replaced RNNs
- [ ] Read: Tokenization
- [ ] Can explain: why `gpt-4o` and `claude-sonnet-4-5` tokenize the same text differently
- [ ] Read: Context Windows
- [ ] Can explain: what happens when context is full
- [ ] Read: How LLMs Generate Text
- [ ] Can set temperature correctly for classification vs creative tasks
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions (can answer without looking)

**Part 2 — Prompt Engineering**
- [ ] Read: Zero-Shot and Few-Shot
- [ ] Built: a few-shot prompt that produces consistent structured output
- [ ] Read: Chain-of-Thought
- [ ] Tested: zero-shot CoT vs no CoT on a math problem
- [ ] Read: Structured Output
- [ ] Can extract: JSON reliably using function calling
- [ ] Read: System Prompts
- [ ] Built: a system prompt with persona, scope, and format constraints
- [ ] Read: Advanced Prompt Patterns
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 02 — LLM APIs + Embeddings

**Part 1 — OpenAI & Anthropic APIs**
- [ ] Made: an authenticated OpenAI API call
- [ ] Made: an authenticated Anthropic API call
- [ ] Read: Vision and Multimodal
- [ ] Read: Tool Use with OpenAI — completed the 4-step tool call cycle
- [ ] Read: Anthropic Messages API — understand differences from OpenAI
- [ ] Read: Cost and Rate Limits
- [ ] Can estimate cost of a call before sending it
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — Embeddings & Semantic Search**
- [ ] Read: What Are Embeddings
- [ ] Read: Embedding Models
- [ ] Ran: `text-embedding-3-small` on sample texts
- [ ] Implemented: cosine similarity from scratch
- [ ] Built: an in-memory semantic search engine
- [ ] Read: Semantic Search Pipeline
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 03 — RAG Basics + Vector Databases

**Part 1 — RAG Basics**
- [ ] Read: What Is RAG
- [ ] Can draw: the RAG architecture from memory
- [ ] Read: Chunking Strategies
- [ ] Implemented: at least two chunking strategies and compared them
- [ ] Read: Retrieval and Augmentation
- [ ] Built: a complete RAG pipeline (PDF → chunk → embed → retrieve → generate)
- [ ] Read: RAG vs Fine-Tuning — can make the decision for a given scenario
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — Vector Databases**
- [ ] Set up: ChromaDB locally, created a collection, added documents
- [ ] Queried: with metadata filters
- [ ] Read: Pinecone — understand serverless vs pod, namespaces
- [ ] Read: Qdrant — understand payload filtering
- [ ] Read: Hybrid Search — can explain RRF
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 04 — Evaluation + Responsible AI

**Part 1 — LLM Evaluation**
- [ ] Read: Evaluation Overview
- [ ] Set up: RAGAS on your Day 03 RAG pipeline
- [ ] Got: faithfulness and answer relevancy scores on ≥ 10 test cases
- [ ] Read: Hallucination and Faithfulness
- [ ] Read: Relevance Metrics
- [ ] Read: Human Evaluations — understand inter-rater reliability
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — Responsible AI & Safety**
- [ ] Read: Jailbreaks and Prompt Injection
- [ ] Implemented: basic input validation / injection detection
- [ ] Read: Guardrails
- [ ] Read: Content Filtering — used the OpenAI Moderation API
- [ ] Read: Bias and Fairness
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 05 — Hugging Face + Local LLMs

**Part 1 — Hugging Face Ecosystem**
- [ ] Navigated: Hugging Face Hub — found a model relevant to your interests
- [ ] Ran: a `pipeline()` inference call locally
- [ ] Called: the Inference API for a hosted model
- [ ] Read: Spaces — understand how to deploy a Gradio demo
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — Local LLMs**
- [ ] Installed: Ollama and pulled a model
- [ ] Made: an API call to Ollama using the OpenAI-compatible endpoint
- [ ] Read: llama.cpp — understand GGUF format
- [ ] Read: Quantization — understand Q4 vs Q8 vs NF4
- [ ] Made: the local vs API decision for one real scenario
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Week 01 Assignment
- [ ] Task 1: API calls to both OpenAI and Anthropic
- [ ] Task 2: Semantic similarity tool with embeddings
- [ ] Task 3: Minimal RAG pipeline with ChromaDB
- [ ] Task 4: RAGAS evaluation with ≥ 10 test cases
- [ ] Task 5: Structured extraction with function calling
- [ ] Submitted: GitHub repository URL

---

## Week 02 — Building LLM Applications

### Day 01 — LangChain + Advanced RAG

**Part 1 — LangChain Fundamentals**
- [ ] Read: LangChain Overview
- [ ] Built: a `ChatPromptTemplate` + LLM + output parser chain
- [ ] Implemented: conversation memory with `RunnableWithMessageHistory`
- [ ] Built: an LCEL pipeline with the pipe operator
- [ ] Called: `.ainvoke()` and `.astream()` on a chain
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — Advanced RAG**
- [ ] Implemented: CrossEncoder reranking on your RAG pipeline
- [ ] Measured: Recall@3 before and after reranking
- [ ] Read: HyDE — can explain when it helps
- [ ] Implemented: multi-query retrieval
- [ ] Read: Contextual Compression
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 02 — Fine-Tuning + Function Calling

**Part 1 — Fine-Tuning**
- [ ] Read: Fine-Tuning Overview — can make the fine-tune vs RAG vs prompting decision
- [ ] Read: LoRA and QLoRA — understand rank, alpha, NF4
- [ ] Set up: a QLoRA training pipeline (even if you ran only a few steps)
- [ ] Read: Training Data — understand chat template format
- [ ] Read: When to Fine-Tune
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — Function Calling & Tool Use**
- [ ] Implemented: the complete 4-step OpenAI tool use cycle from scratch
- [ ] Implemented: Anthropic tool_use (note: `block.input` is a dict, not a string)
- [ ] Built: a structured extraction pipeline with Pydantic and `with_structured_output`
- [ ] Tested: parallel tool calls
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 03 — AI Agents + LangGraph

**Part 1 — AI Agents**
- [ ] Read: What Are AI Agents — can decide when an agent is warranted
- [ ] Implemented: a basic ReAct loop from scratch (no framework)
- [ ] Read: Planning Strategies
- [ ] Built: a tool-calling agent with at least 2 tools
- [ ] Implemented: sliding window memory
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — LangGraph**
- [ ] Built: a minimal StateGraph with one node
- [ ] Used: `Annotated[list, operator.add]` as a reducer
- [ ] Implemented: conditional routing with `add_conditional_edges`
- [ ] Added: `MemorySaver` checkpointing
- [ ] Used: `interrupt_before` for a human-in-the-loop pattern
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 04 — LLMOps + Deployment

**Part 1 — LLMOps**
- [ ] Instrumented: an LLM call with structured logging (latency, tokens, cost)
- [ ] Set up: LangSmith tracing with `@traceable`
- [ ] Implemented: cost tracking with tiktoken
- [ ] Read: Latency Optimization — applied at least one technique
- [ ] Read: Observability
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

**Part 2 — Deployment**
- [ ] Built: a FastAPI endpoint that calls an LLM
- [ ] Used: `AsyncOpenAI` (not sync client) in the endpoint
- [ ] Implemented: SSE streaming with `StreamingResponse`
- [ ] Added: `X-Accel-Buffering: no` header
- [ ] Implemented: exact-match cache with SHA256 key
- [ ] Read: Serverless Deployment
- [ ] Completed: Practice Exercises
- [ ] Reviewed: Interview Questions

---

### Day 05 — Capstone + Interview Prep

**Capstone Project**
- [ ] Chose: one of the three capstone options
- [ ] Completed: architecture design doc
- [ ] Built: working FastAPI endpoint (not just a script)
- [ ] Implemented: async patterns correctly throughout
- [ ] Added: error handling for API failures and invalid input
- [ ] Ran: quantitative evaluation with ≥ 10 test cases
- [ ] Wrote: README with setup instructions and eval results

**Stretch goals (3 points each)**
- [ ] Streaming response via SSE
- [ ] LangSmith tracing enabled
- [ ] Deployed on Fly.io, Modal, or Railway
- [ ] RAGAS or LLM-as-judge evaluation
- [ ] Cost tracking per request

**Interview Preparation**
- [ ] Read: Interview Preparation Overview
- [ ] Practiced: technical Q&A out loud (not just reading)
- [ ] Completed: at least one system design problem
- [ ] Ran: Mock Interview Simulator (all 10 questions)
- [ ] Updated: resume with course project bullet
- [ ] Pushed: capstone project to GitHub with clean README

---

### Week 02 Assignment
- [ ] Working FastAPI endpoint
- [ ] Async patterns used correctly
- [ ] 4+ course components integrated
- [ ] Quantitative evaluation with ≥ 10 test cases + results in README
- [ ] Submission checklist completed
- [ ] Submitted: GitHub repository URL

---

## Final Checklist Before Job Applications

- [ ] GitHub has ≥ 2 pinned LLM projects with clear READMEs and eval results
- [ ] Resume has ≥ 1 bullet with specific metrics ("faithfulness 0.87", "p95 latency 1.2s")
- [ ] Can explain RAG architecture without notes in under 2 minutes
- [ ] Can walk through the function calling cycle from scratch
- [ ] Can describe LoRA — what it trains, what it freezes, why it's memory-efficient
- [ ] Practiced ≥ 2 system design problems out loud
- [ ] Know the answer to: "Why would you use AsyncOpenAI over the sync client?"
