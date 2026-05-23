# Interview Questions — Local LLMs

---

## Q1: What is GGUF and how does it differ from safetensors?

??? "Show answer"
    **GGUF** (GPT-Generated Unified Format) is the file format used by llama.cpp for CPU-optimized inference. It stores model weights, tokenizer data, and metadata in a single self-contained file, with quantized weights (Q4, Q8, etc.) that enable efficient CPU inference.

    **safetensors** is a general-purpose model weight format developed by Hugging Face. It stores weights in full precision (usually float32 or float16) and is optimized for GPU loading via memory mapping.

    **Key differences:**

    | Feature | GGUF | safetensors |
    |---------|------|-------------|
    | Primary use | CPU inference (llama.cpp, Ollama) | GPU training/inference (transformers, vLLM) |
    | Quantization | Built-in (Q2–Q8) | Separate step (GPTQ, AWQ, BnB) |
    | Tokenizer | Embedded in file | Separate file |
    | Loading speed | Fast (memory-mapped) | Fast (memory-mapped) |
    | Ecosystem | llama.cpp, Ollama | Hugging Face transformers |

    You would use GGUF when running models locally without a GPU, or when you want a single-file distribution. You would use safetensors when using the `transformers` library for GPU inference or fine-tuning.

---

## Q2: Explain partial GPU layer offloading in llama.cpp. When would you use it?

??? "Show answer"
    A transformer model consists of a fixed number of layers (e.g., 32 layers for a 7B model). In llama.cpp, `n_gpu_layers` controls how many of these layers are computed on the GPU versus the CPU.

    ```python
    # GPU has 6 GB VRAM; 7B Q4 model needs 4.1 GB
    # Set n_gpu_layers based on how much fits in GPU
    llm = Llama(model_path="model.gguf", n_gpu_layers=28)  # 28 of 32 layers on GPU
    ```

    Each layer takes roughly equal memory. If your GPU has 6 GB VRAM and the full model needs 4.1 GB, you can fit about 28 of 32 layers (28/32 × 4.1 ≈ 3.6 GB). The remaining 4 layers run on CPU.

    **Speed impact:** Tokens-per-second scales roughly with the fraction of layers on GPU. With 28/32 layers on GPU, you get ~85–90% of full-GPU speed. Even partial offloading dramatically outperforms CPU-only.

    **When to use it:**
    - Your GPU has less VRAM than the quantized model size
    - You want faster inference than CPU-only but can't afford a larger GPU
    - Running on Apple Silicon (unified memory — all layers can go on GPU)

    **Practical example:** RTX 4060 Ti (16 GB) can run Llama 3.1 70B Q4 (40 GB) with n_gpu_layers=20, getting some GPU acceleration even though the full model doesn't fit.

---

## Q3: What is the difference between Q4_K_M and Q4_0 quantization in GGUF?

??? "Show answer"
    Both use 4 bits per weight, but they differ in how the quantization values are chosen:

    **Q4_0 (simple quantization):**
    - Divides each 32-weight block by its maximum absolute value
    - Maps to 16 values: {-8, -7, ..., 0, ..., 7, 8} scaled to the block range
    - Fast and simple but uses suboptimal quantization points

    **Q4_K_M (k-means quantization, medium variant):**
    - Uses k-means clustering to find the 16 quantization values that minimize error for each block
    - The cluster centroids are data-dependent and better represent the weight distribution
    - Typically 1–2% better quality than Q4_0 at the same bit width
    - Slightly larger file (~5% bigger than Q4_0) due to storing the centroids

    **Recommendation:** Always use Q4_K_M over Q4_0 — the quality improvement is free (only ~5% larger file). The "K" variants (K_S, K_M, K_L) use progressively more bits for attention and feed-forward layers, with K_M being the default sweet spot.

---

## Q4: When would you recommend a hybrid local+API architecture?

??? "Show answer"
    A hybrid architecture routes requests to local or API models based on their characteristics. It makes sense when:

    **Routing by complexity:** Simple factual lookups don't need frontier-model quality. Route "what is X?" to a local 7B model; route complex multi-step reasoning to GPT-4o. You can reduce API costs by 60–80% while maintaining quality for hard queries.

    **Routing by data sensitivity:** Route queries containing PII or confidential data to local inference; route public-domain queries to the API. This satisfies compliance requirements without going fully local.

    **Routing by cost threshold:** Set a monthly API budget. When the budget is exceeded, automatically fall back to local inference for lower-priority tasks.

    **Implementation pattern:**
    ```python
    local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def route(complexity: str, has_pii: bool) -> tuple[OpenAI, str]:
        if has_pii or complexity == "simple":
            return local_client, "llama3.1:8b"
        return api_client, "gpt-4o"
    ```

    The key challenge: the complexity classifier itself must be fast and accurate. A slow router negates the latency benefit of local inference.

---

## Q5: What are the main reasons a 7B local model might produce worse output than GPT-4o-mini even though they have similar parameter counts?

??? "Show answer"
    GPT-4o-mini is not literally a 7B model — OpenAI has not disclosed its size, but it's likely a much larger model (potentially Mixture of Experts with many more total parameters) or trained on significantly more data with better techniques.

    Even comparing two true 7B models, quality differences come from:

    **Training data quality and quantity:** Frontier API models are trained on vastly more curated data. Llama 3.1 8B was trained on 15 trillion tokens — impressive, but GPT-4o-mini likely saw more curated, higher-quality data.

    **RLHF quality:** The instruction-tuning step (RLHF/DPO) determines how well the model follows instructions. Frontier models have had significantly more human preference data collected.

    **Training compute:** A 7B model trained with 10× more compute (using better optimizers, learning rate schedules, curriculum) outperforms the same architecture trained with less.

    **Quantization loss:** A local model in Q4 quantization loses ~3% of quality vs full precision. The API serves full-precision (or F16) models.

    **Practical impact:** On simple factual questions, code generation for common patterns, and summarization, a well-tuned 8B model (Llama 3.1, Mistral 7B) is close to GPT-4o-mini quality. On complex multi-step reasoning, long-context tasks, and nuanced instruction following, the gap is noticeably larger.

---

## Q6: How would you set up a local LLM inference server for a team of 5 developers?

??? "Show answer"
    The simplest approach for a small team: run Ollama as a shared server on a single machine.

    **Hardware:** One machine with a GPU (RTX 4090 or A100) accessible on your local network.

    **Setup:**
    ```bash
    # On the server machine
    OLLAMA_HOST=0.0.0.0:11434 ollama serve   # Listen on all interfaces
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    ```

    **Each developer's machine:**
    ```python
    from openai import OpenAI
    # Point to the shared server
    client = OpenAI(base_url="http://gpu-server.local:11434/v1", api_key="ollama")
    ```

    **Scaling considerations:**
    - Ollama queues concurrent requests — it doesn't serve multiple users in parallel on the same model
    - With 5 developers, request queuing is usually fine for development use
    - If you need true parallel inference, use vLLM (serves multiple requests simultaneously via continuous batching) on a multi-GPU server
    - Add authentication via a reverse proxy (nginx) if the server is on a shared network

    **Monitoring:** Ollama exposes a `/api/ps` endpoint showing running models and memory usage. Wrap this in a simple dashboard for the team to check availability.

---

[[06-practice-exercises]] | [[../../Week-02/index]]
