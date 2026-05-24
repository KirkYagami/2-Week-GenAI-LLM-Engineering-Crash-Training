# Glossary

Key terms in LLM engineering. Each entry has a brief definition and a link to the course note where the concept is covered in depth.

---

## A

**Attention mechanism** — The core operation in transformers. Each token computes a weighted sum of all other tokens in the context, where weights (attention scores) represent relevance. Enables the model to "focus" on relevant parts of the input regardless of distance.
→ [Transformers and Attention](Week-01/Day-01-Part-1-How-LLMs-Work/01-transformers-and-attention.md)

**Autoregressive generation** — How LLMs generate text: predict one token at a time, append it to the context, and repeat. Each token depends on all previous tokens.
→ [How LLMs Generate Text](Week-01/Day-01-Part-1-How-LLMs-Work/04-how-llms-generate-text.md)

---

## B

**BM25** — A sparse retrieval algorithm based on term frequency and inverse document frequency. Excels at exact keyword matching; used in hybrid search alongside dense vector retrieval.
→ [Hybrid Search](Week-01/Day-03-Part-2-Vector-Databases/06-hybrid-search.md)

**BitsAndBytes (bnb)** — A library for quantizing model weights to 4-bit or 8-bit precision during training and inference, enabling large models to fit on smaller GPUs.
→ [LoRA and QLoRA](Week-02/Day-02-Part-1-Fine-Tuning/02-lora-and-qlora.md)

---

## C

**Chain** — A fixed sequence of LLM calls and operations. The execution path is predetermined at build time. Contrast with *agent*.
→ [LangChain Overview](Week-02/Day-01-Part-1-LangChain-Fundamentals/01-langchain-overview.md)

**Chain-of-Thought (CoT)** — A prompting technique where the model is instructed to generate intermediate reasoning steps before producing the final answer. Improves performance on multi-step reasoning tasks.
→ [Chain-of-Thought](Week-01/Day-01-Part-2-Prompt-Engineering/02-chain-of-thought.md)

**Chunking** — Splitting a document into smaller text segments before embedding. Chunk size and overlap are key hyperparameters that affect retrieval quality.
→ [Chunking Strategies](Week-01/Day-03-Part-1-RAG-Basics/02-chunking-strategies.md)

**ChromaDB** — An open-source vector database for storing and querying embeddings locally or in a server. Commonly used for RAG prototyping and small-scale production.
→ [ChromaDB](Week-01/Day-03-Part-2-Vector-Databases/02-chromadb.md)

**Constitutional AI (CAI)** — Anthropic's approach to training AI systems using AI-generated feedback based on a set of principles, rather than human labels for every example.
→ [Responsible AI and Safety](Week-01/Day-04-Part-2-Responsible-AI-and-Safety/00-agenda.md)

**Context window** — The maximum number of tokens an LLM can process in a single call (input + output combined). Determines how much text the model can "see" at once.
→ [Context Windows](Week-01/Day-01-Part-1-How-LLMs-Work/03-context-windows.md)

**Cosine similarity** — A measure of the angle between two vectors. Returns 1.0 for identical direction, 0 for orthogonal, -1 for opposite. Used to compare embedding vectors in semantic search.
→ [Cosine Similarity](Week-01/Day-02-Part-2-Embeddings-and-Semantic-Search/03-cosine-similarity.md)

**Cross-encoder** — A reranking model that processes the query and a candidate document together (with full attention between them), producing a relevance score. More accurate than bi-encoders but O(n) per query.
→ [Reranking](Week-02/Day-01-Part-2-Advanced-RAG/01-reranking.md)

---

## D

**Dense retrieval** — Embedding-based retrieval where query and documents are encoded into continuous vector space and compared by cosine similarity. Captures semantic meaning. Contrast with *sparse retrieval*.
→ [Vector Search](Week-01/Day-02-Part-2-Embeddings-and-Semantic-Search/04-vector-search.md)

---

## E

**Embedding** — A dense vector representation of text that encodes semantic meaning. Similar texts have similar embeddings (small cosine distance). Produced by embedding models like `text-embedding-3-small`.
→ [What Are Embeddings](Week-01/Day-02-Part-2-Embeddings-and-Semantic-Search/01-what-are-embeddings.md)

**Embedding model** — A model trained to produce embeddings. Distinct from a language model — embedding models produce fixed-size vectors, not text.
→ [Embedding Models](Week-01/Day-02-Part-2-Embeddings-and-Semantic-Search/02-embedding-models.md)

---

## F

**Faithfulness** — A RAGAS metric measuring whether the generated answer contains only claims that are supported by the retrieved context. A faithfulness score of 1.0 means every claim in the answer can be traced to the context.
→ [RAGAS Framework](Week-01/Day-04-Part-1-LLM-Evaluation/02-ragas-framework.md)

**Few-shot prompting** — Including 2–8 input/output examples in the prompt before the actual query. Helps the model understand the expected format and behaviour.
→ [Zero-Shot and Few-Shot](Week-01/Day-01-Part-2-Prompt-Engineering/01-zero-shot-and-few-shot.md)

**Fine-tuning** — Updating a pretrained model's weights on a task-specific dataset to improve performance on that task. Contrast with RAG (which adds knowledge at inference time) and prompting (which doesn't change weights).
→ [Fine-Tuning Overview](Week-02/Day-02-Part-1-Fine-Tuning/01-fine-tuning-overview.md)

**Function calling** — An API feature that lets you define JSON Schema tool definitions; the model returns structured tool invocations instead of free text. Also called "tool use."
→ [Function Calling Overview](Week-02/Day-02-Part-2-Function-Calling-and-Tool-Use/01-function-calling-overview.md)

---

## G

**GGUF** — A file format for quantized LLM weights, used by `llama.cpp` for CPU inference. Replaces the older GGML format.
→ [llama.cpp](Week-01/Day-05-Part-2-Local-LLMs/03-llama-cpp.md)

**Grounding** — Anchoring model outputs to a specific, verifiable source (documents, database records). RAG is a grounding technique.
→ [What Is RAG](Week-01/Day-03-Part-1-RAG-Basics/01-what-is-rag.md)

**Guardrails** — Input and output checks that enforce safety and quality constraints on LLM pipelines. Can be rule-based, classifier-based, or model-based.
→ [Guardrails](Week-01/Day-04-Part-2-Responsible-AI-and-Safety/02-guardrails.md)

---

## H

**Hallucination** — When a model generates plausible-sounding but factually incorrect or unsupported content. The primary failure mode RAG is designed to mitigate.
→ [Hallucination and Faithfulness](Week-01/Day-04-Part-1-LLM-Evaluation/03-hallucination-and-faithfulness.md)

**HNSW (Hierarchical Navigable Small World)** — The graph-based index algorithm used by most vector databases for approximate nearest neighbour search. Provides fast retrieval with small accuracy loss vs brute force.
→ [Indexing and Filtering](Week-01/Day-03-Part-2-Vector-Databases/05-indexing-and-filtering.md)

**Human-in-the-loop** — A design pattern where a human reviews and approves agent actions before they are executed. Implemented in LangGraph via `interrupt_before`.
→ [Multi-Agent Orchestration](Week-02/Day-03-Part-2-LangGraph/05-multi-agent-orchestration.md)

**HyDE (Hypothetical Document Embeddings)** — A retrieval technique that generates a hypothetical answer to the query and embeds that instead of the raw query. Bridges the query-document embedding gap.
→ [HyDE](Week-02/Day-01-Part-2-Advanced-RAG/02-hyde.md)

---

## I

**In-context learning** — The ability of LLMs to perform new tasks given only examples in the prompt, without any weight updates. Demonstrated by GPT-3.
→ [Zero-Shot and Few-Shot](Week-01/Day-01-Part-2-Prompt-Engineering/01-zero-shot-and-few-shot.md)

---

## J

**JSON mode** — An API parameter (`response_format={"type": "json_object"}`) that guarantees the model's output is valid JSON. Does not enforce a specific schema.
→ [Structured Output](Week-01/Day-01-Part-2-Prompt-Engineering/03-structured-output.md)

---

## K

**KV cache** — A performance optimization in transformer inference. Key and value matrices from previous tokens are cached so they don't need to be recomputed for each new token. Reduces inference cost for long contexts.
→ [Context Windows](Week-01/Day-01-Part-1-How-LLMs-Work/03-context-windows.md)

---

## L

**LangChain** — A framework for building LLM applications with composable chains, LCEL pipelines, memory, and integrations.
→ [LangChain Fundamentals](Week-02/Day-01-Part-1-LangChain-Fundamentals/00-agenda.md)

**LangGraph** — A library for building stateful, graph-structured LLM applications with explicit state management, conditional routing, and checkpointing.
→ [LangGraph](Week-02/Day-03-Part-2-LangGraph/00-agenda.md)

**LangSmith** — An observability platform for LLM applications. Captures traces, datasets, and evaluations for chains, agents, and RAG pipelines.
→ [LangSmith](Week-02/Day-04-Part-1-LLMOps/02-langsmith.md)

**LCEL (LangChain Expression Language)** — A declarative syntax for composing LangChain components using the pipe operator (`|`). Supports async, streaming, and parallel execution.
→ [LCEL](Week-02/Day-01-Part-1-LangChain-Fundamentals/04-lcel.md)

**LLM-as-judge** — Using a capable LLM to evaluate the outputs of another LLM on a rubric (e.g., faithfulness 1–5, relevance 1–5). Enables automated evaluation without human annotation.
→ [Human Evaluations](Week-01/Day-04-Part-1-LLM-Evaluation/05-human-evals.md)

**LLMOps** — The operational practices for deploying, monitoring, and maintaining LLM-powered applications in production.
→ [LLMOps](Week-02/Day-04-Part-1-LLMOps/00-agenda.md)

**LoRA (Low-Rank Adaptation)** — A PEFT method that freezes base model weights and adds trainable low-rank decomposition matrices to specific layers. Enables efficient fine-tuning with a small fraction of the parameters.
→ [LoRA and QLoRA](Week-02/Day-02-Part-1-Fine-Tuning/02-lora-and-qlora.md)

---

## M

**MTEB (Massive Text Embedding Benchmark)** — A benchmark evaluating embedding models across 56 tasks (retrieval, clustering, classification, etc.). The standard leaderboard for choosing embedding models.
→ [Embedding Models](Week-01/Day-02-Part-2-Embeddings-and-Semantic-Search/02-embedding-models.md)

**Multi-head attention** — Running multiple attention operations in parallel, each with different learned weight matrices. The outputs are concatenated and projected. Allows the model to attend to different aspects of the input simultaneously.
→ [Transformers and Attention](Week-01/Day-01-Part-1-How-LLMs-Work/01-transformers-and-attention.md)

---

## N

**NF4 (Normal Float 4)** — A 4-bit quantization format used in QLoRA. Bin boundaries are spaced to match the normal distribution of pretrained weights, making it more accurate than standard 4-bit integer quantization.
→ [LoRA and QLoRA](Week-02/Day-02-Part-1-Fine-Tuning/02-lora-and-qlora.md)

---

## O

**Ollama** — A tool for running open-source LLMs locally with an OpenAI-compatible REST API. Manages model downloads and serves inference.
→ [Ollama](Week-01/Day-05-Part-2-Local-LLMs/02-ollama.md)

---

## P

**PEFT (Parameter-Efficient Fine-Tuning)** — A family of methods for fine-tuning large models by training only a small subset of parameters. LoRA is the most widely used PEFT method.
→ [PEFT](Week-02/Day-02-Part-1-Fine-Tuning/03-peft.md)

**Perplexity** — A measure of how well a language model predicts a held-out text. Lower perplexity = the model is less surprised by the text. Used to compare models on domain fit.
→ [Evaluation](Week-02/Day-02-Part-1-Fine-Tuning/06-practice-exercises.md)

**Prompt caching** — An Anthropic feature that caches large prompt prefixes (system prompts, context) for 5 minutes, charging ~10% of normal input token price on cache hits.
→ [Anthropic Messages API](Week-01/Day-02-Part-1-OpenAI-and-Anthropic-APIs/04-anthropic-messages-api.md)

**Prompt injection** — An attack where user input contains instructions that override or extend the system prompt, causing the model to behave in unintended ways.
→ [Jailbreaks and Prompt Injection](Week-01/Day-04-Part-2-Responsible-AI-and-Safety/01-jailbreaks-and-prompt-injection.md)

---

## Q

**QLoRA** — Quantized LoRA. Combines 4-bit NF4 quantization of the base model with LoRA adapter training in bfloat16. Enables fine-tuning 7B–70B models on a single GPU.
→ [LoRA and QLoRA](Week-02/Day-02-Part-1-Fine-Tuning/02-lora-and-qlora.md)

**Quantization** — Reducing the numerical precision of model weights (e.g., from float32 to int4) to decrease memory requirements and speed up inference, at the cost of some accuracy.
→ [Quantization](Week-01/Day-05-Part-2-Local-LLMs/04-quantization.md)

---

## R

**RAG (Retrieval-Augmented Generation)** — A technique that retrieves relevant documents from an external knowledge base and includes them in the prompt before generation. Reduces hallucination and enables knowledge-grounded responses.
→ [What Is RAG](Week-01/Day-03-Part-1-RAG-Basics/01-what-is-rag.md)

**RAGAS** — A framework for evaluating RAG pipelines using automated metrics: faithfulness, answer relevancy, context recall, and context precision.
→ [RAGAS Framework](Week-01/Day-04-Part-1-LLM-Evaluation/02-ragas-framework.md)

**ReAct** — A prompting pattern that interleaves Reasoning (Thought) and Acting (Action/Observation) in an agent loop. The conceptual basis for most LLM agent frameworks.
→ [The ReAct Loop](Week-02/Day-03-Part-1-AI-Agents/02-react-loop.md)

**Recall@k** — The fraction of relevant documents that appear in the top-k retrieval results. A primary metric for evaluating retrieval quality.
→ [Evaluation Overview](Week-01/Day-04-Part-1-LLM-Evaluation/01-evaluation-overview.md)

**Reducer** — In LangGraph, a function that defines how to merge a new value with the existing state value for a key. `operator.add` appends to lists; default (no reducer) replaces.
→ [Stateful Graphs](Week-02/Day-03-Part-2-LangGraph/03-stateful-graphs.md)

**Reranking** — A post-retrieval step that scores retrieved documents for relevance using a cross-encoder model. Improves retrieval precision at the cost of added latency.
→ [Reranking](Week-02/Day-01-Part-2-Advanced-RAG/01-reranking.md)

**RLHF (Reinforcement Learning from Human Feedback)** — A training technique that uses human preference judgments to train a reward model, which is then used to fine-tune the LLM via RL. Used to align models with human preferences.
→ [Responsible AI and Safety](Week-01/Day-04-Part-2-Responsible-AI-and-Safety/00-agenda.md)

---

## S

**Self-consistency** — A prompting technique that samples the model multiple times with temperature > 0 and takes a majority vote on the final answer. Improves accuracy at the cost of higher token usage.
→ [Chain-of-Thought](Week-01/Day-01-Part-2-Prompt-Engineering/02-chain-of-thought.md)

**Semantic cache** — A cache that stores LLM responses indexed by embeddings. On a new query, checks cosine similarity to cached queries; returns a cached response if similarity exceeds a threshold.
→ [Caching Strategies](Week-02/Day-04-Part-2-Deployment/05-caching-strategies.md)

**SFTTrainer** — The TRL library's supervised fine-tuning trainer. Handles dataset formatting, gradient accumulation, and PEFT integration.
→ [PEFT](Week-02/Day-02-Part-1-Fine-Tuning/03-peft.md)

**Sparse retrieval** — Retrieval based on exact token matching (BM25, TF-IDF). Strong for queries with specific identifiers; weak for semantic similarity.
→ [Hybrid Search](Week-01/Day-03-Part-2-Vector-Databases/06-hybrid-search.md)

**SSE (Server-Sent Events)** — A one-way HTTP streaming protocol where the server pushes `data: {content}\n\n` events to the client. Used for streaming LLM token output.
→ [Streaming Responses](Week-02/Day-04-Part-2-Deployment/02-streaming-responses.md)

**StateGraph** — The core LangGraph class. Defines nodes (functions), edges (transitions), and state schema. Compiled into a runnable graph.
→ [Nodes and Edges](Week-02/Day-03-Part-2-LangGraph/02-nodes-and-edges.md)

**System prompt** — Instructions passed to the model before the conversation begins. Defines persona, scope, and format constraints. Processed with higher priority than user messages in most models.
→ [System Prompts](Week-01/Day-01-Part-2-Prompt-Engineering/04-system-prompts.md)

---

## T

**Temperature** — A parameter that controls the randomness of token sampling. `temperature=0` = deterministic (greedy), `temperature=1` = sample from full distribution.
→ [How LLMs Generate Text](Week-01/Day-01-Part-1-How-LLMs-Work/04-how-llms-generate-text.md)

**tiktoken** — OpenAI's tokenizer library. Used to count tokens before sending requests (for cost estimation) and to implement token-aware chunking.
→ [Tokenization](Week-01/Day-01-Part-1-How-LLMs-Work/02-tokenization.md)

**Token** — The basic unit of text that LLMs process. Roughly 0.75 words in English. Pricing, context limits, and rate limits are all measured in tokens.
→ [Tokenization](Week-01/Day-01-Part-1-How-LLMs-Work/02-tokenization.md)

**Tool use** — See *function calling*.

**top_p (nucleus sampling)** — Sample only from the smallest set of tokens whose cumulative probability reaches `top_p`. Alternative to temperature for controlling randomness.
→ [How LLMs Generate Text](Week-01/Day-01-Part-1-How-LLMs-Work/04-how-llms-generate-text.md)

**Transformer** — The neural network architecture underlying all modern LLMs. Uses self-attention to process all tokens in parallel, enabling long-range dependencies.
→ [Transformers and Attention](Week-01/Day-01-Part-1-How-LLMs-Work/01-transformers-and-attention.md)

---

## V

**Vector database** — A database optimised for storing and querying high-dimensional embedding vectors. Key capabilities: approximate nearest neighbour search, metadata filtering, CRUD operations.
→ [Vector DB Overview](Week-01/Day-03-Part-2-Vector-Databases/01-vector-db-overview.md)

**vLLM** — A high-throughput LLM serving library with PagedAttention for efficient KV cache management. Used for self-hosted production inference.
→ [Local LLMs](Week-01/Day-05-Part-2-Local-LLMs/00-agenda.md)

---

## Z

**Zero-shot prompting** — Giving the model only instructions, with no examples. Works when the task is within the model's training distribution and the instructions are sufficiently clear.
→ [Zero-Shot and Few-Shot](Week-01/Day-01-Part-2-Prompt-Engineering/01-zero-shot-and-few-shot.md)
