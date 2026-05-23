# Training Data Preparation

The most common cause of poor fine-tuning results isn't the model or the hyperparameters — it's the data. Garbage in, garbage out applies with extreme force to LLMs: the model will learn *exactly* what your data shows, including its inconsistencies, errors, and biases.

## Learning objectives

- Format datasets in JSONL for SFT
- Apply label masking so loss only computes on completions
- Assess data quality: diversity, consistency, length distribution
- Handle data imbalance and deduplication
- Know how much data you need for common task types

---

## The JSONL format

The standard format for SFT datasets is JSONL (JSON Lines) — one training example per line:

```python
import json

# Format 1: Alpaca-style (instruction, input, output)
alpaca_examples = [
    {
        "instruction": "Classify the sentiment of this review.",
        "input": "The product broke after two days. Terrible quality.",
        "output": "negative"
    },
    {
        "instruction": "Classify the sentiment of this review.",
        "input": "Amazing! Exactly what I needed.",
        "output": "positive"
    },
]

# Format 2: ShareGPT-style (messages list — preferred for chat models)
sharegpt_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are a sentiment classifier. Respond with: positive, negative, or neutral."},
            {"role": "user", "content": "The product broke after two days. Terrible quality."},
            {"role": "assistant", "content": "negative"}
        ]
    },
]

# Write to JSONL
with open("train.jsonl", "w") as f:
    for ex in sharegpt_examples:
        f.write(json.dumps(ex) + "\n")
```

---

## Label masking: only train on the completion

In instruction fine-tuning, you don't want the model to learn to predict the prompt — only the response. Label masking sets prompt tokens to -100 (ignored in loss computation).

```python
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def format_and_mask(example: dict) -> dict:
    """
    Format a messages-style example and mask prompt tokens in labels.
    Only the assistant turn contributes to the loss.
    """
    # Format full prompt including response
    full_text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )

    # Format prompt only (no response)
    prompt_messages = [m for m in example["messages"] if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize both
    full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]

    # Build labels: -100 for prompt, actual IDs for completion
    labels = full_tokens.clone()
    labels[:len(prompt_tokens)] = -100

    return {
        "input_ids": full_tokens,
        "labels": labels,
        "attention_mask": torch.ones_like(full_tokens),
    }

# Test
example = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}
result = format_and_mask(example)
print("input_ids length:", len(result["input_ids"]))
print("labels (non-masked portion):", result["labels"][result["labels"] != -100])
```

> [!warning] SFTTrainer handles masking automatically
> If you use `SFTTrainer` with `DataCollatorForCompletionOnlyLM`, masking is handled automatically. Only implement manual masking if you're writing a custom training loop.

---

## Data quality checklist

```python
from datasets import Dataset
from collections import Counter
import re

def audit_dataset(examples: list[dict]) -> dict:
    """Quick quality audit for SFT datasets."""
    lengths = []
    seen = set()
    duplicates = 0
    empty_outputs = 0

    for ex in examples:
        # Get output text
        if "messages" in ex:
            output = next((m["content"] for m in ex["messages"] if m["role"] == "assistant"), "")
        else:
            output = ex.get("output", "")

        if not output.strip():
            empty_outputs += 1
            continue

        # Check for duplicates
        key = output.strip().lower()
        if key in seen:
            duplicates += 1
        seen.add(key)

        lengths.append(len(output.split()))

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    return {
        "total": len(examples),
        "empty_outputs": empty_outputs,
        "duplicates": duplicates,
        "avg_output_words": round(avg_len, 1),
        "min_output_words": min(lengths) if lengths else 0,
        "max_output_words": max(lengths) if lengths else 0,
    }

# Sample audit
examples = [
    {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "Answer one"}]},
    {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "Answer one"}]},  # duplicate output
    {"messages": [{"role": "user", "content": "Q3"}, {"role": "assistant", "content": ""}]},            # empty
]
report = audit_dataset(examples)
print(report)
# {'total': 3, 'empty_outputs': 1, 'duplicates': 1, 'avg_output_words': 2.0, 'min': 2, 'max': 2}
```

---

## How much data do you need?

```python
DATA_REQUIREMENTS = {
    "Classification (2–5 classes)": {
        "minimum": "50–100 examples per class",
        "good": "200–500 per class",
        "notes": "Balance classes within 2:1 ratio"
    },
    "Named entity extraction": {
        "minimum": "200–500 diverse sentences",
        "good": "1,000–5,000",
        "notes": "Entity diversity matters more than raw count"
    },
    "Style / tone transfer": {
        "minimum": "100–300 before/after pairs",
        "good": "500–1,000",
        "notes": "Quality >> quantity; inconsistent pairs hurt badly"
    },
    "Domain-specific Q&A": {
        "minimum": "500–1,000 Q&A pairs",
        "good": "2,000–10,000",
        "notes": "Cover edge cases, not just common queries"
    },
    "Code generation": {
        "minimum": "1,000 function-level examples",
        "good": "10,000+",
        "notes": "Include tests; code must be executable"
    },
}

for task, reqs in DATA_REQUIREMENTS.items():
    print(f"\n{task}")
    print(f"  Minimum: {reqs['minimum']}")
    print(f"  Good:    {reqs['good']}")
    print(f"  Note:    {reqs['notes']}")
```

---

## Handling class imbalance

```python
from datasets import Dataset, concatenate_datasets
from collections import Counter

def balance_dataset(examples: list[dict], label_field: str = "output", max_ratio: float = 3.0) -> list[dict]:
    """Downsample majority class to max_ratio × minority class size."""
    by_label: dict[str, list] = {}
    for ex in examples:
        label = ex[label_field]
        by_label.setdefault(label, []).append(ex)

    counts = {k: len(v) for k, v in by_label.items()}
    min_count = min(counts.values())
    max_allowed = int(min_count * max_ratio)

    balanced = []
    for label, items in by_label.items():
        balanced.extend(items[:max_allowed])

    print(f"Original: {Counter(ex[label_field] for ex in examples)}")
    print(f"Balanced: {Counter(ex[label_field] for ex in balanced)}")
    return balanced

# Example
raw = (
    [{"output": "positive"}] * 1000 +
    [{"output": "negative"}] * 200 +
    [{"output": "neutral"}] * 50
)
balanced = balance_dataset(raw, max_ratio=2.0)
```

---

## Generating synthetic training data

When labeled data is scarce, use a strong model to generate synthetic examples:

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_synthetic_examples(task_description: str, n: int = 20) -> list[dict]:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Generate {n} diverse training examples for: {task_description}
Return JSON array: [{{"input": "...", "output": "..."}}]
Ensure diverse inputs covering edge cases."""
        }],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("examples", data.get("data", []))

examples = generate_synthetic_examples(
    "Classify customer support tickets as: billing, technical, account, or general",
    n=10
)
for ex in examples[:3]:
    print(f"Input: {ex['input'][:60]}")
    print(f"Output: {ex['output']}\n")
```

> [!warning] Verify synthetic data quality
> Synthetic data from GPT-4o is high-quality but can be homogeneous — the model generates the "typical" example, not the edge case. Always review a sample manually and supplement with real edge cases.

---

[[03-peft]] | [[05-when-to-fine-tune]]
