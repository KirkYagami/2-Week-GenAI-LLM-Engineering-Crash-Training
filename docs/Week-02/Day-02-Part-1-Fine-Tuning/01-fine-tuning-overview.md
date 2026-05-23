# Fine-Tuning Overview

A pre-trained LLM knows how to generate text. Fine-tuning teaches it *what kind* of text to generate for your specific task. The difference between a model that sometimes does what you want and one that reliably does what you need is often a few hundred well-chosen training examples.

## Learning objectives

- Distinguish full fine-tuning, PEFT, and inference-time techniques (RAG, prompting)
- Understand what gradient updates actually change in a transformer
- Know the cost difference between full fine-tuning and LoRA
- Identify which technique addresses which failure mode

---

## What fine-tuning changes

Pre-training teaches a model the statistical structure of language. Fine-tuning specializes it:

```
Pre-training:   "Complete this text in any plausible way"
Instruction FT: "Follow instructions in the format shown"
Task FT:        "Extract entities from customer support tickets"
Style FT:       "Write in the formal tone of legal documents"
```

Mechanically, fine-tuning runs gradient descent on labeled examples, updating model weights to minimize loss on your specific task. The backbone stays the same; the weights shift toward your distribution.

---

## The four options

```
                    Data needed    Cost      Latency    Customization
─────────────────────────────────────────────────────────────────────
Prompting           0 examples    $0        0ms extra  Low
RAG                 0 examples    Storage   +50ms      Medium (domain)
PEFT (LoRA/QLoRA)   100–10K       $10–200   0ms        High
Full fine-tuning    10K–100K      $500–10K  0ms        Highest
```

**Prompting:** Free, instant, but limited by the model's pre-existing behaviors. Works for general tasks the model already does well.

**RAG:** Injects up-to-date knowledge at query time without changing weights. Best when the problem is *what the model knows*, not *how it behaves*.

**PEFT (LoRA/QLoRA):** Updates a small fraction of parameters (1–3% of total). Cost is 10–100x lower than full fine-tuning. Suitable for most task adaptation.

**Full fine-tuning:** Updates all parameters. Requires large datasets and significant compute. Justified only when PEFT isn't sufficient or you're training from scratch.

---

## The training loop (simplified)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# One gradient step (conceptually)
def training_step(batch: dict) -> float:
    input_ids = batch["input_ids"].to(model.device)
    labels = batch["labels"].to(model.device)

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

For instruction fine-tuning, `labels` mask out the prompt tokens — you only compute loss on the completion tokens. The model learns to generate the right output, not to predict the prompt.

---

## Instruction fine-tuning format

Most modern fine-tuning uses a chat template. The model learns to follow the structure:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful customer support agent."},
    {"role": "user", "content": "My order hasn't arrived."},
    {"role": "assistant", "content": "I'm sorry to hear that. Can you share your order number?"}
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
print(formatted)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful customer support agent.<|eot_id|>...
```

The formatted string becomes one training example. Thousands of such examples, run through gradient descent, shift the model toward your target behavior.

---

## Supervised fine-tuning vs RLHF

| Stage | What it optimizes | Data type | Used for |
|-------|------------------|-----------|----------|
| Supervised FT (SFT) | Minimize loss on human-written outputs | Input → output pairs | Task adaptation |
| Reward Modeling (RM) | Predict human preference score | Ranked pairs | RLHF stage 1 |
| RLHF / PPO | Maximize reward model score | Model outputs + scores | Alignment |
| DPO | Maximize preferred over rejected | Preferred/rejected pairs | Alignment (simpler) |

For task-specific fine-tuning (classification, extraction, style), SFT on labeled examples is almost always the right approach. RLHF and DPO are for alignment — teaching models to be helpful and harmless.

> [!success] Most task fine-tuning uses SFT only
> Unless you're training a general assistant, supervised fine-tuning on high-quality labeled examples is sufficient. RLHF adds significant complexity and requires a reliable reward signal — hard to get right.

---

## Common misconceptions

> [!warning] Fine-tuning doesn't add knowledge
> Fine-tuning changes *behavior*, not *knowledge*. If the model doesn't know a fact before fine-tuning, it won't know it after. Use RAG to inject knowledge; use fine-tuning to change output format, style, or task-following behavior.

> [!warning] More data isn't always better
> 100 high-quality, diverse examples often outperform 10,000 noisy examples. Fine-tuning amplifies the patterns in your data — bad data teaches the model bad patterns.

---

[[00-agenda]] | [[02-lora-and-qlora]]
