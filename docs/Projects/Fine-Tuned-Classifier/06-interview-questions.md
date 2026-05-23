# Interview Questions — Fine-Tuned Classifier

---

**Q1: You fine-tuned a 0.5B model. Your zero-shot gpt-4o-mini baseline gets 0.90 F1. The fine-tuned model gets 0.88 F1. Was fine-tuning worth it?**

??? "Show answer"
    Depends on the use case. On accuracy alone: no, zero-shot wins by 0.02. But consider:
    
    - **Cost**: gpt-4o-mini costs $0.15/1M input tokens. Your 0.5B model running on a $0.30/hr Fly.io instance costs ~$0.0001 per inference at 200 inferences/hour. At 10,000 classifications/day, fine-tuning breaks even in days.
    - **Latency**: a local 0.5B model can respond in < 100ms on GPU vs. 300–800ms for gpt-4o-mini API round trips.
    - **Privacy**: if the reviews contain PII, sending them to an external API is a compliance risk. A local model keeps data in-house.
    - **Improvement ceiling**: 0.88 was trained on 50-100 examples. With 500+ examples and hyperparameter tuning, the fine-tuned model may surpass the baseline.
    
    The answer is context-dependent. Present the tradeoffs, don't just compare the numbers.

---

**Q2: What is label masking and why does it matter for fine-tuning a classifier?**

??? "Show answer"
    During causal LM training, the loss is computed as the average cross-entropy over all tokens in the sequence. Without label masking, the model is penalized for not predicting the system prompt and user message accurately — wasting gradient updates on tokens it never needs to generate.
    
    Label masking sets the target token IDs to -100 for all non-assistant tokens. PyTorch's `CrossEntropyLoss` ignores positions where the target is -100. The gradient only flows through the label tokens ("positive", "negative", "neutral").
    
    In practice: without label masking, the model learns to reproduce the prompt. With label masking, all gradient signal is concentrated on the one-word label, which leads to faster convergence and higher classification accuracy for a given number of training steps.

---

**Q3: What does `r=16` mean in `LoraConfig(r=16, ...)`?**

??? "Show answer"
    `r` is the rank of the low-rank matrices in LoRA. For a weight matrix W of shape (d_in × d_out), LoRA replaces the update ΔW with two matrices: A (d_in × r) and B (r × d_out), where r << d_in and r << d_out.
    
    - `r=4` — minimal parameters, lowest memory cost, works for simple tasks
    - `r=16` — good default, balances expressivity and parameter efficiency
    - `r=64` — higher capacity, needed for complex tasks or small datasets
    
    For a 0.5B model's attention layers (d=512 or d=1024), r=16 means: instead of updating 512×512=262,144 parameters per layer, you update 512×16 + 16×512 = 16,384 parameters — about 6% of the original. `lora_alpha=32` scales the update magnitude: the effective delta is `(alpha/r) * AB = (32/16) * AB = 2 * AB`.

---

**Q4: How would you add a fourth label ("mixed") to the classifier after initial training?**

??? "Show answer"
    You can't extend the label set by continuing training on the existing checkpoint — the model has already learned to associate "positive", "negative", and "neutral" with the task. Adding a new label requires:
    
    1. **Add mixed examples** to the training set — at least 50 examples labeled "mixed".
    2. **Update the system prompt** to include "mixed" as a valid output: "Respond with exactly one word: positive, negative, neutral, or mixed."
    3. **Re-train from scratch** on the full dataset (all four labels). Don't continue from the 3-class checkpoint — the model hasn't learned to generate "mixed" as a valid response and will resist it.
    4. **Re-evaluate** against the ground truth test set which should also include mixed examples.
    
    This is a known limitation of instruction-tuned classifiers: changing the label space requires re-training, unlike traditional classifiers where you can sometimes add a class head.

---

[[05-deployment]]
