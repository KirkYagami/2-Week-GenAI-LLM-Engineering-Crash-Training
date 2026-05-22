# Interview Questions — How LLMs Work

These are the questions you'll actually get asked in ML engineering, MLOps, and LLM application roles. Each answer is written at the level expected in a senior engineer interview — not textbook, not too long.

---

## Q1: What is the transformer architecture and why did it replace RNNs?

??? "Show answer"
    The transformer is a neural network architecture based entirely on the self-attention mechanism, introduced in "Attention Is All You Need" (Vaswani et al., 2017). It processes all tokens in the input sequence simultaneously, rather than one at a time.

    RNNs had two fatal flaws at scale:
    1. **Sequential processing** — each step depends on the previous one, so training cannot be parallelized across GPUs
    2. **Vanishing gradients** — information from early tokens is compressed into a fixed-size hidden state and degrades over long sequences

    Transformers solve both: attention computes relationships between *any* two tokens directly, regardless of distance, and the entire sequence is processed in parallel. This is why transformers scale with compute in a way RNNs never could.

---

## Q2: Explain self-attention. What are Q, K, and V?

??? "Show answer"
    Self-attention computes, for each token, a weighted average of all other tokens in the sequence — where the weights are determined by relevance.

    Three learned linear projections are applied to each token embedding:
    - **Q (Query):** "What am I looking for?"
    - **K (Key):** "What do I offer to other queries?"
    - **V (Value):** "What is my actual content?"

    The attention score between token `i` and token `j` is:
    ```
    attention(i, j) = softmax( Q_i · K_j / √d_k )
    ```

    The output for token `i` is the weighted sum of all Value vectors. The `√d_k` scaling prevents dot products from growing too large in high dimensions, which would push softmax into regions with near-zero gradients.

    **Practical implication:** The bank in "I deposited money at the bank" will attend more strongly to "money" and "deposited" than to "river", correctly disambiguating the meaning.

---

## Q3: What is the difference between encoder-only, decoder-only, and encoder-decoder models? When would you use each?

??? "Show answer"
    | Architecture | Training Objective | Best For |
    |---|---|---|
    | Encoder-only (BERT) | Masked LM — predict randomly masked tokens | Embeddings, classification, NER, semantic search |
    | Decoder-only (GPT-4o, Claude) | Causal LM — predict the next token | Text generation, chat, reasoning, code |
    | Encoder-decoder (T5, BART) | Seq2seq — encode full input, decode output | Translation, summarization, question answering |

    In practice, most LLM applications in 2025 use decoder-only models (GPT-4o, Claude, Llama 3) because they generalize better to diverse tasks through prompting and RLHF. Encoder-only models are still preferred when you need dense vector representations (embeddings) for semantic search.

---

## Q4: What is tokenization and why does the choice of tokenizer matter?

??? "Show answer"
    Tokenization converts raw text into a sequence of integer IDs that the model processes. Modern LLMs use Byte Pair Encoding (BPE): start with individual characters, then iteratively merge the most frequent adjacent pairs until reaching the target vocabulary size.

    Why the choice matters:
    1. **Cost** — you pay per token. Non-English languages tokenize less efficiently, costing 2–4× more per semantic unit.
    2. **Arithmetic** — the number "1234567" might tokenize as ["123", "456", "7"], explaining why LLMs struggle with multi-digit arithmetic.
    3. **Context utilization** — a model with a 100K token context can fit less content if your tokenizer is inefficient.
    4. **Vocabulary size** — GPT-4o uses `o200k_base` (200K vocab), GPT-4 uses `cl100k_base` (100K vocab). Larger vocab = fewer tokens needed for the same text.

    Always count tokens with the actual tokenizer before sending large documents to the API.

---

## Q5: What is the context window? What is the "lost in the middle" problem?

??? "Show answer"
    The context window is the maximum number of tokens a model can process in a single forward pass — input plus output combined. Modern frontier models range from 128K (GPT-4o) to 1M tokens (Claude Sonnet 4.6, Gemini 3).

    The "lost in the middle" problem (Liu et al., 2023): models perform significantly worse at recalling information placed in the middle of a long context compared to information at the beginning or end. Accuracy on middle-context facts can drop by 20–30% even with models that nominally "support" long contexts.

    Practical mitigation:
    - Place the most critical information first or last in the context
    - Use retrieval to select the 3–10 most relevant chunks rather than feeding everything
    - Apply reranking to ensure the top-ranked chunks end up at the beginning

---

## Q6: What is the difference between temperature, top-k, and top-p? Which should you use?

??? "Show answer"
    All three control the sampling distribution after the model computes logits:

    - **Temperature:** Scales logits before softmax. Low temperature (< 1) makes the distribution sharper — the model picks from fewer high-probability tokens. High temperature (> 1) flattens the distribution, making low-probability tokens more likely.
    - **Top-k:** Sample only from the k highest-probability tokens. Simple but doesn't adapt to distribution shape.
    - **Top-p (nucleus):** Sample from the smallest set of tokens whose cumulative probability ≥ p. Adapts to the distribution — picks 1–2 tokens when confident, 20–30 when uncertain.

    **Recommendation:**
    - Use `temperature=0` (or 0.0–0.2) for extraction, classification, structured output
    - Use `temperature=0.7–1.0` for generation; pair with `top_p=0.9–0.95`
    - Don't combine aggressive temperature with aggressive top-p — they interact non-linearly
    - Reasoning models (o3, o4-mini) ignore temperature; use `reasoning_effort` instead

---

## Q7: What is a KV cache and why does it matter for production LLM serving?

??? "Show answer"
    During autoregressive generation, each new token needs the Key and Value matrices of all previous tokens to compute attention. Without caching, these would be recomputed for every new token — O(n²) work per output token.

    The KV cache stores K and V matrices for all previously processed tokens. New tokens only compute attention with the cached values, reducing per-token generation to O(n) work.

    Why it matters in production:
    - **Memory:** The KV cache consumes significant GPU memory (can be 10–30 GB for a 70B model with a 128K context). This limits batch size and therefore throughput.
    - **Cost:** Cloud providers charge for compute. A full KV cache hit (via prompt caching in Anthropic/OpenAI) can reduce input token cost by ~90%.
    - **Latency:** Time to first token (TTFT) is dominated by prefill (processing the prompt). Generation speed is dominated by KV cache bandwidth.

    Architecture innovations like GQA (Grouped-Query Attention) and MLA (Multi-Head Latent Attention) directly target KV cache reduction.

---

## Q8: What is the difference between pretraining and fine-tuning? What is RLHF?

??? "Show answer"
    **Pretraining:** The model learns to predict the next token on a massive corpus (trillions of tokens from the web, books, code). This teaches language, facts, and reasoning patterns. It's unsupervised — no human labels needed.

    **Fine-tuning (SFT):** After pretraining, the model is trained on a smaller, high-quality dataset of (prompt, ideal response) pairs. This teaches the model to follow instructions and produce useful outputs. Supervised — requires human-curated data.

    **RLHF (Reinforcement Learning from Human Feedback):** Human raters compare model outputs and rank them by preference. A reward model is trained to predict human preferences. The LLM is then trained with RL (typically PPO) to maximize the reward model's score. This is what turns a capable-but-unhelpful model into ChatGPT or Claude.

    The typical training pipeline: Pretraining → SFT → RLHF. More recent approaches include DPO (Direct Preference Optimization) which skips the reward model step.

---

## Q9: Why do LLMs hallucinate? Can it be completely fixed?

??? "Show answer"
    LLMs hallucinate because they are trained to generate *plausible* text, not *accurate* text. The training objective — predict the next token — does not distinguish between factual and counterfactual content.

    Mechanistically, hallucination occurs when:
    1. The model's parametric memory contains incorrect or outdated facts
    2. The model over-generalizes patterns ("The capital of France is X" → "The capital of Y is Z")
    3. The model confabulates to fill gaps rather than saying "I don't know"
    4. Long contexts cause attention to diffuse, making retrieved facts compete with parametric memory

    Partial mitigations (none eliminate it):
    - RAG: ground the model in retrieved source documents
    - Structured output: constrain the model to validated fields
    - Verification: use a second model call to check the first response
    - Calibrated uncertainty: prompt the model to say "I'm not sure" when confidence is low

    No, it cannot be completely fixed with current architectures. Hallucination is an inherent consequence of statistical language modeling.

---

## Q10: How many parameters does GPT-4 have? Why does parameter count matter less than it used to?

??? "Show answer"
    GPT-4's parameter count was never officially disclosed, but estimates range from ~1.7 trillion (as a Mixture of Experts) to 220 billion (as a dense model). OpenAI has not confirmed.

    Parameter count matters less than it used to because:
    1. **Training data quality dominates.** Llama 3 8B, trained on 15 trillion tokens of curated data, outperforms GPT-3 175B on most benchmarks.
    2. **Architecture efficiency.** MoE models like Mixtral 8×7B have 46B total parameters but only 12B active per forward pass.
    3. **Post-training alignment.** RLHF and SFT have a larger practical effect on output quality than adding more parameters.
    4. **Quantization.** A 70B model at 4-bit quantization runs on a single A100 and performs comparably to its full-precision counterpart.

    The scaling law (Chinchilla, 2022) showed that most LLMs were undertrained relative to their size — you get more performance from more training data than from more parameters, up to the compute-optimal ratio.

---

[[05-practice-exercises]] | [[Week-01/Day-01-Part-2-Prompt-Engineering/00-agenda]]
