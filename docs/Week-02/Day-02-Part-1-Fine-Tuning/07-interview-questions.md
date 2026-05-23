# Interview Questions — Fine-Tuning

---

**Q1: What does LoRA stand for, and why does its low-rank decomposition work for fine-tuning?**

??? "Show answer"
    **LoRA** stands for **Low-Rank Adaptation**. It works because the weight updates needed for task adaptation have *low intrinsic rank* — meaning the useful change can be captured by a small subspace rather than a full-rank matrix.

    Formally: instead of learning a full ΔW (e.g., 4096×4096 = 16M parameters), LoRA learns A (4096×r) and B (r×4096), where r is 4–64. For r=16, that's 131K parameters — 128x fewer.

    This works empirically because tasks don't require changing the entire model — they require shifting the model's behavior in a constrained direction. The pre-trained model already has the general capability; fine-tuning nudges it toward the target distribution. That nudge lives in a low-dimensional subspace.

---

**Q2: What is the difference between `lora_alpha` and `r` in LoraConfig, and how do they interact?**

??? "Show answer"
    - **`r` (rank):** Determines the number of parameters. Higher r = more parameters = more adaptation capacity.
    - **`lora_alpha`:** A scaling factor. The effective LoRA contribution is scaled by `alpha / r`. It's equivalent to a learning rate multiplier for the adapter.

    Common convention: set `alpha = 2 × r`. This gives a scaling factor of 2, which empirically provides a good balance.

    If you increase r (more capacity), also increase alpha proportionally to keep the same effective scale. If you find training is unstable, try reducing alpha while keeping r fixed.

    Practical example: `r=16, alpha=32` → scale = 2.0. If you double the rank to `r=32` without changing alpha, scale becomes 1.0, effectively halving the adapter's contribution — not what you want.

---

**Q3: What is the purpose of `prepare_model_for_kbit_training()` in QLoRA?**

??? "Show answer"
    When you load a model in 4-bit quantization with `BitsAndBytesConfig(load_in_4bit=True)`, the model weights are stored in quantized format that isn't differentiable — you can't backpropagate through them directly.

    `prepare_model_for_kbit_training()` does three things:
    1. Freezes all base model parameters (they won't be updated)
    2. Casts layer normalization layers to float32 (they must remain in higher precision for stable training)
    3. Enables gradient checkpointing compatibility with quantized layers

    Without this call, training on a quantized model either fails with dtype errors or produces incorrect gradients.

---

**Q4: Explain label masking in instruction fine-tuning. Why do we set prompt tokens to -100?**

??? "Show answer"
    In supervised fine-tuning, the model takes the full prompt + completion as input and is trained to predict each token given the previous ones (autoregressive language modeling).

    If you compute loss on the prompt tokens too, the model learns to predict the prompt — which is already fixed and deterministic. This wastes the training signal and can cause the model to generate the prompt format as part of its answers.

    Setting prompt token labels to **-100** tells PyTorch's `CrossEntropyLoss` to ignore those positions when computing the loss. Only the completion tokens contribute gradients. The model learns: "given this prompt, generate this specific completion."

    In practice: `SFTTrainer` with `DataCollatorForCompletionOnlyLM` handles this automatically by finding the response template token and masking everything before it.

---

**Q5: When would you choose fine-tuning over RAG, and when would you combine them?**

??? "Show answer"
    **Choose fine-tuning over RAG when:**
    - The problem is *behavior*, not *knowledge* (output format, style, task-following)
    - Retrieval latency is unacceptable (fine-tuned model has no retrieval step)
    - You're replacing an expensive API call with a cheaper fine-tuned small model
    - The task requires consistent output across thousands of edge cases that prompting can't handle reliably

    **Choose RAG over fine-tuning when:**
    - The model lacks factual knowledge (fine-tuning doesn't reliably add knowledge)
    - The knowledge changes frequently (retraining is expensive; updating a vector DB is cheap)
    - You need to cite sources
    - You have no labeled training data

    **Combine them when:**
    - RAG retrieves relevant context (solves the knowledge problem)
    - A fine-tuned model acts as the reader/generator (solves the behavior/format problem)
    - Example: fine-tune a smaller model to generate structured JSON answers from retrieved context, replacing a larger API model in the pipeline

---

**Q6: How do you detect overfitting during LoRA fine-tuning, and how do you fix it?**

??? "Show answer"
    **Detection:** You need a held-out evaluation set (typically 10–20% of your data). Track both training loss and eval loss during training. Overfitting shows as:
    - Training loss continues to decrease
    - Eval loss plateaus or increases
    - The gap between train and eval loss grows each epoch

    If you have no eval set (very small dataset), overfitting shows as the model memorizing training inputs — it generates the exact training completions rather than generalizing.

    **Fixes (in order of impact):**
    1. **Reduce epochs:** Most LoRA fine-tuning needs 1–5 epochs; running 10–20 epochs almost always overfits
    2. **Increase `lora_dropout`:** From 0.05 to 0.10 or 0.15
    3. **Reduce rank:** Lower r = fewer parameters = less memorization capacity
    4. **Add more training data:** More diverse examples reduce overfitting
    5. **Reduce learning rate:** Helps when loss oscillates rather than steadily diverging

---

**Q7: What does `merge_and_unload()` do, and when should you use it vs keeping adapters separate?**

??? "Show answer"
    `merge_and_unload()` adds the LoRA adapter weights (A×B) back into the base model's weight matrices (W' = W + A×B) and returns a standard `AutoModelForCausalLM` with no PEFT dependency. The result:
    - Runs at the same speed as the base model (no adapter overhead at inference)
    - Is fully portable — no PEFT library needed at inference time
    - Larger disk footprint (full model size, not just adapter)

    **Use merged model when:**
    - Deploying to production and you want minimal dependencies
    - Serving only one fine-tuned variant
    - Optimizing inference speed (every operation matters)

    **Keep adapters separate when:**
    - Running multiple fine-tuned variants of the same base model (store one base + N small adapters)
    - Frequently updating the adapter (retrain adapter, keep base frozen)
    - Disk space is constrained (adapter files are 50–150 MB vs 14 GB for the merged 7B model)

---

[[06-practice-exercises]]
