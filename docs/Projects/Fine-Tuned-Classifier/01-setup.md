# Setup and Environment — Fine-Tuned Classifier

## Project structure

```
fine-tuned-classifier/
├── prepare_data.py     ← Create and format training dataset
├── train.py            ← QLoRA fine-tuning script
├── merge.py            ← Merge LoRA adapter into base model
├── eval.py             ← Compare fine-tuned vs zero-shot
├── app.py              ← FastAPI inference endpoint
├── requirements.txt
├── .env.example
└── data/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## Environment setup

```bash
# GPU environment (Colab or local GPU)
pip install transformers==4.45.0 peft==0.13.0 trl==0.11.4 bitsandbytes==0.44.0
pip install datasets==3.0.1 accelerate==1.0.1

# For the inference server (CPU-only OK)
pip install fastapi uvicorn openai python-dotenv
```

```bash
# .env.example
OPENAI_API_KEY=sk-...       # For baseline comparison
MODEL_PATH=./merged-model   # Path to merged fine-tuned model
```

## Dataset preparation

```python
# prepare_data.py
import json
import random

# Sample training data — replace with your domain-specific data
LABELED_EXAMPLES = [
    ("This product is absolutely amazing, exactly what I needed!", "positive"),
    ("Works as described, nothing special but does the job.", "neutral"),
    ("Complete waste of money. Broke after two days.", "negative"),
    ("Best purchase I've made all year. Highly recommend!", "positive"),
    ("Arrived damaged and customer support was unhelpful.", "negative"),
    ("Average product, average price. Does what it says.", "neutral"),
    ("Exceeded all my expectations. Five stars!", "positive"),
    ("Not terrible but not great either. Would probably not buy again.", "neutral"),
    ("Absolute garbage. Stay far away from this product.", "negative"),
    ("Solid quality for the price. Would buy again.", "positive"),
    # Add 40+ more examples for a meaningful fine-tuning run
]

SYSTEM_PROMPT = (
    "Classify the sentiment of the following product review. "
    "Respond with exactly one word: positive, negative, or neutral."
)

def format_example(text: str, label: str) -> dict:
    """Format as a chat conversation with the label as the assistant response."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": label},
        ]
    }

def prepare_dataset(examples: list[tuple], train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    for split_name, split_data in splits.items():
        with open(f"data/{split_name}.jsonl", "w") as f:
            for text, label in split_data:
                f.write(json.dumps(format_example(text, label)) + "\n")
        print(f"{split_name}: {len(split_data)} examples → data/{split_name}.jsonl")

if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    prepare_dataset(LABELED_EXAMPLES)
```

```bash
python prepare_data.py
# train: 7 examples → data/train.jsonl
# val: 2 examples → data/val.jsonl
# test: 1 examples → data/test.jsonl
```

> [!warning] 10 examples is not enough for fine-tuning
> The sample data above is for illustration only. A meaningful fine-tuning run requires at least 50–100 labeled examples per class (150–300 total). Use a public dataset like the Amazon product review dataset or collect domain-specific examples.

---

[[README]] | [[02-implementation]]
