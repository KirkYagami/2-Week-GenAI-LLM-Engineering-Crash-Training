# Evaluation — Fine-Tuned Classifier

## The comparison you're trying to answer

> Does fine-tuning a small model outperform prompting a large model on this task?

Run both models on the same held-out test set and compare F1 scores. If fine-tuned wins: fine-tuning was worth it. If zero-shot wins: the task wasn't narrow enough or your training data was insufficient.

## Evaluation script

```python
# eval.py
import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI
from sklearn.metrics import classification_report, f1_score
from dotenv import load_dotenv

load_dotenv()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "Classify the sentiment of the following product review. "
    "Respond with exactly one word: positive, negative, or neutral."
)

def load_test_set(path: str = "data/test.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

def extract_label(messages: list[dict]) -> str:
    """Extract ground truth label from the assistant message."""
    for msg in messages:
        if msg["role"] == "assistant":
            return msg["content"].strip().lower()
    return "unknown"

def extract_text(messages: list[dict]) -> str:
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"]
    return ""

# ---- Zero-shot baseline ----
async def predict_zero_shot(text: str) -> str:
    resp = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    raw = resp.choices[0].message.content.strip().lower()
    return next((l for l in ["positive", "negative", "neutral"] if l in raw), "unknown")

async def run_zero_shot_eval(test_set: list[dict]) -> tuple[list[str], list[str]]:
    texts = [extract_text(ex["messages"]) for ex in test_set]
    labels = [extract_label(ex["messages"]) for ex in test_set]
    predictions = await asyncio.gather(*[predict_zero_shot(t) for t in texts])
    return labels, list(predictions)

# ---- Fine-tuned model ----
def run_finetuned_eval(test_set: list[dict], base_url: str = "http://localhost:8000") -> tuple[list[str], list[str]]:
    labels = [extract_label(ex["messages"]) for ex in test_set]
    predictions = []
    with httpx.Client(timeout=30.0) as client:
        for ex in test_set:
            text = extract_text(ex["messages"])
            resp = client.post(f"{base_url}/classify", json={"text": text})
            pred = resp.json()["label"] if resp.status_code == 200 else "unknown"
            predictions.append(pred)
    return labels, predictions

# ---- Compare ----
def compare(zero_shot_labels, zero_shot_preds, finetuned_labels, finetuned_preds) -> None:
    zs_f1 = f1_score(zero_shot_labels, zero_shot_preds, average="macro", zero_division=0)
    ft_f1 = f1_score(finetuned_labels, finetuned_preds, average="macro", zero_division=0)

    print("\n=== Zero-Shot (gpt-4o-mini) ===")
    print(classification_report(zero_shot_labels, zero_shot_preds, zero_division=0))
    print(f"Macro F1: {zs_f1:.3f}")

    print("\n=== Fine-Tuned (Qwen2-0.5B-Instruct + QLoRA) ===")
    print(classification_report(finetuned_labels, finetuned_preds, zero_division=0))
    print(f"Macro F1: {ft_f1:.3f}")

    print(f"\n=== Summary ===")
    print(f"Zero-shot F1:  {zs_f1:.3f}")
    print(f"Fine-tuned F1: {ft_f1:.3f}")
    winner = "Fine-tuned" if ft_f1 > zs_f1 else "Zero-shot"
    delta = abs(ft_f1 - zs_f1)
    print(f"Winner: {winner} by {delta:.3f}")

async def main():
    test_set = load_test_set()
    print(f"Test set: {len(test_set)} examples")

    print("\nRunning zero-shot evaluation...")
    zs_labels, zs_preds = await run_zero_shot_eval(test_set)

    print("\nRunning fine-tuned evaluation (start the app.py server first)...")
    ft_labels, ft_preds = run_finetuned_eval(test_set)

    compare(zs_labels, zs_preds, ft_labels, ft_preds)

if __name__ == "__main__":
    asyncio.run(main())
```

## Expected results

On a well-prepared 150-example dataset (50 per class):

| Model | Macro F1 | Notes |
|-------|----------|-------|
| Zero-shot gpt-4o-mini | 0.88–0.92 | Strong baseline for sentiment |
| 5-shot gpt-4o-mini | 0.90–0.94 | Marginal improvement |
| Fine-tuned Qwen2-0.5B | 0.85–0.95 | Depends heavily on data quality |

Fine-tuning a 0.5B model often matches gpt-4o-mini zero-shot on simple classification — and is much cheaper at inference time.

> [!info] When fine-tuning beats zero-shot prompting
> Fine-tuning wins on: domain-specific jargon not in the training data ("churned", "MRR", "CAC" in a SaaS context), consistent output format requirements, very high volume (inference cost savings), and latency-sensitive applications.
>
> Zero-shot prompting wins on: tasks requiring broad knowledge, small datasets (< 50 examples per class), and when the task changes frequently.

---

[[03-advanced-features]] | [[05-deployment]]
