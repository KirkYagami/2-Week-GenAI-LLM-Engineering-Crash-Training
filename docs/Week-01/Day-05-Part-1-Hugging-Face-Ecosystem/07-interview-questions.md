# Interview Questions — Hugging Face Ecosystem

---

## Q1: What is the difference between the Hugging Face Inference API and an Inference Endpoint?

??? "Show answer"
    Both allow you to call models hosted on Hugging Face infrastructure, but they differ in isolation, latency, and cost:

    **Inference API (serverless):**
    - Uses a shared pool of GPU workers
    - Free tier: ~300 requests/hour; pay-per-call for higher volume
    - Cold start latency: up to 60 seconds when model is idle
    - Supports only public Hub models
    - No SLA — best effort availability

    **Inference Endpoints:**
    - Dedicated compute instance (you choose GPU type and region)
    - Always warm — no cold start
    - Supports private models and custom containers
    - Billed by the hour (~$0.40–$4/hour depending on GPU)
    - Uptime SLA available

    **When to use which:**
    - Inference API: development, prototyping, testing model quality, low-volume demos
    - Endpoints: production applications, latency-sensitive use cases, private models, > 1000 requests/hour

---

## Q2: What does `apply_chat_template()` do and why should you always use it?

??? "Show answer"
    `apply_chat_template()` converts a list of `{"role": ..., "content": ...}` message dicts into the model-specific string format expected by that model's tokenizer.

    Different instruction-tuned models use different formats:
    - Llama 3 uses `<|start_header_id|>user<|end_header_id|>...<|eot_id|>`
    - Mistral uses `[INST]...[/INST]`
    - ChatML format uses `<|im_start|>user\n...<|im_end|>`

    If you manually format the prompt incorrectly (wrong tokens, missing delimiters), the model will produce lower quality output without raising any error — it will just interpret the malformed prompt as best it can.

    `apply_chat_template()` reads the correct format from the tokenizer's `tokenizer_config.json` and applies it automatically. This means the same code works across model families just by changing the model ID.

    ```python
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,           # Return string, not token IDs
        add_generation_prompt=True  # Add the assistant turn start token
    )
    ```

    Always set `add_generation_prompt=True` — without it, the model doesn't know it's supposed to start generating.

---

## Q3: How does `BitsAndBytesConfig` 4-bit quantization work, and what is the memory reduction?

??? "Show answer"
    `BitsAndBytesConfig` with `load_in_4bit=True` uses NF4 (NormalFloat4) quantization, introduced by the QLoRA paper. It stores model weights in 4-bit integers instead of 16-bit floats.

    **Memory reduction:** Float16 uses 2 bytes per parameter. NF4 uses 0.5 bytes per parameter — a 4× reduction.

    | Model | Float16 | NF4 4-bit |
    |-------|---------|-----------|
    | 7B params | 14 GB | ~3.5 GB |
    | 13B params | 26 GB | ~6.5 GB |
    | 70B params | 140 GB | ~35 GB |

    **Double quantization** (`bnb_4bit_use_double_quant=True`) additionally quantizes the quantization constants, saving ~0.4 GB per 7B parameters.

    **Compute dtype** (`bnb_4bit_compute_dtype=torch.bfloat16`): Weights are stored in 4-bit but computations are done in bfloat16. This maintains numeric stability while minimizing storage.

    **Quality loss:** Benchmarks show < 1% degradation on most tasks compared to float16. For fine-tuning with LoRA (QLoRA), the trained adapter compensates for quantization noise and the final model often matches float16 quality.

---

## Q4: What is the `datasets` library streaming mode and when would you use it?

??? "Show answer"
    `streaming=True` in `load_dataset()` enables lazy loading: data is downloaded and processed in small chunks on demand, never fully loaded into memory.

    ```python
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    for batch in dataset.iter(batch_size=32):
        process(batch)  # Only 32 examples in memory at once
    ```

    **Use streaming when:**
    - Dataset is larger than available RAM (e.g., C4 ~305 GB, The Pile ~825 GB)
    - You only need a sample (e.g., `dataset.take(10000)` without downloading everything)
    - You're doing one-pass processing (embedding, filtering, statistics)

    **Don't use streaming when:**
    - You need random access by index (streaming only supports sequential access)
    - You're doing multiple passes over the data (training requires re-iteration)
    - You need `.shuffle()` with a large buffer (shuffling is limited in streaming mode)

    For training, download the full dataset with `streaming=False`, process it with `.map()` (which caches results), and use a `DataLoader`. Reserve streaming for EDA and one-shot processing tasks.

---

## Q5: How would you find the best open-source embedding model for your specific domain?

??? "Show answer"
    Three-step process:

    **Step 1 — Use MTEB as a starting point**
    MTEB (Massive Text Embedding Benchmark) ranks hundreds of embedding models across 56 tasks: classification, clustering, retrieval, semantic similarity. Filter by `task_type=Retrieval` (most relevant for RAG) and `model_size_MB` to fit your hardware.

    Start with top performers: `BAAI/bge-large-en-v1.5`, `intfloat/e5-large-v2`, `text-embedding-3-small` (OpenAI). These give you a competitive baseline.

    **Step 2 — Build a domain-specific eval set**
    MTEB is general-purpose. Your domain may have different characteristics:
    - Collect 50–100 (query, relevant_doc) pairs from your corpus
    - Run each candidate model on this set
    - Measure Recall@5 and MRR

    **Step 3 — Consider practical tradeoffs**
    - `all-MiniLM-L6-v2` (22M params, 80ms): fast, fits on CPU, good quality for most use cases
    - `bge-large-en-v1.5` (335M params, 800ms on CPU): best quality, needs GPU for production
    - `text-embedding-3-small` (API): no local inference, great quality, $0.02/1M tokens

    For specialized domains (medical, legal, code), look for domain-adapted models: `BAAI/bge-en-icl` (in-context learning), `nomic-ai/nomic-embed-text-v1.5` (long documents), `jinaai/jina-embeddings-v3` (multilingual).

---

## Q6: What are the required files for a valid Hugging Face model repository?

??? "Show answer"
    A minimal valid `transformers` model repository needs:

    **Required:**
    - `config.json` — model architecture and hyperparameters
    - `tokenizer_config.json` — tokenizer settings and chat template
    - Model weights in one of:
      - `pytorch_model.bin` (legacy) or `model.safetensors` (recommended)
      - For large models: sharded `model-00001-of-00004.safetensors` etc.

    **Strongly recommended:**
    - `README.md` (model card with YAML frontmatter)
    - `tokenizer.json` — fast tokenizer implementation
    - `special_tokens_map.json` — BOS, EOS, PAD tokens

    **Optional but useful:**
    - `generation_config.json` — default generation parameters
    - `merges.txt` / `vocab.txt` — for BPE/WordPiece tokenizers
    - `adapter_config.json` / `adapter_model.safetensors` — for LoRA/PEFT models

    **safetensors vs bin:**
    Always prefer `.safetensors` format. It loads 2–3× faster, supports memory mapping (no full load into RAM needed before GPU transfer), and is safer (no arbitrary Python code execution unlike pickle-based `.bin`).

---

[[06-practice-exercises]] | [[../Day-05-Part-2-Local-LLMs/00-agenda]]
