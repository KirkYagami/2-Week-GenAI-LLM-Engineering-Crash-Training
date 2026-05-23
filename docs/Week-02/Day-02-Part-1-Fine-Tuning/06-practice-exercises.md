# Practice Exercises — Fine-Tuning

---

## Exercise 1 — Decision framework (Warm-up)

Given the scenarios below, apply the decision framework and justify your choice: prompting, RAG, or fine-tuning.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCENARIOS = [
    {
        "id": 1,
        "description": "A law firm wants an LLM to summarize court documents using their specific citation format (e.g., '[Plaintiff] v. [Defendant], Case No. XXXX'). Prompting achieves ~70% correct format compliance.",
        "available_examples": 150,
        "query_volume": "500/day",
        "data_changes": False,
    },
    {
        "id": 2,
        "description": "A startup wants to answer questions about their internal product documentation (200 pages, updated weekly).",
        "available_examples": 0,
        "query_volume": "1000/day",
        "data_changes": True,
    },
    {
        "id": 3,
        "description": "An e-commerce company wants to classify product return reasons into 8 categories. GPT-4o achieves 91% accuracy; they need 97%+ with 50ms latency.",
        "available_examples": 2000,
        "query_volume": "50000/day",
        "data_changes": False,
    },
]

def analyze_scenario(scenario: dict) -> str:
    prompt = f"""Given this scenario, recommend ONE technique: prompting, RAG, or fine-tuning.
Justify in 2-3 sentences.

Scenario: {scenario['description']}
Available labeled examples: {scenario['available_examples']}
Query volume: {scenario['query_volume']}
Data changes frequently: {scenario['data_changes']}

Recommendation:"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=150
    )
    return resp.choices[0].message.content

for s in SCENARIOS:
    print(f"\nScenario {s['id']}:")
    print(f"  {s['description'][:80]}...")
    print(f"  Recommendation: {analyze_scenario(s)}")
```

Expected: Scenario 1 → fine-tuning (behavior/format), Scenario 2 → RAG (knowledge, updates), Scenario 3 → fine-tuning + smaller model (latency + cost).

---

## Exercise 2 — LoRA config comparison (Main)

Train two LoRA adapters with different rank configurations on a small sentiment dataset and compare their performance and parameter counts.

```python
import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

MODEL_ID = "microsoft/phi-2"

# Small but realistic sentiment dataset
TRAIN_EXAMPLES = [
    {"text": "### Classify: 'Absolutely love this!' ### Answer: positive"},
    {"text": "### Classify: 'Worst product I have ever used.' ### Answer: negative"},
    {"text": "### Classify: 'Arrived on time, nothing remarkable.' ### Answer: neutral"},
    {"text": "### Classify: 'Exceeded all expectations!' ### Answer: positive"},
    {"text": "### Classify: 'Broke after one week. Terrible.' ### Answer: negative"},
    {"text": "### Classify: 'Does the job. Nothing more.' ### Answer: neutral"},
    {"text": "### Classify: 'Incredible value for money!' ### Answer: positive"},
    {"text": "### Classify: 'Customer service was awful.' ### Answer: negative"},
    {"text": "### Classify: 'Works as advertised.' ### Answer: neutral"},
    {"text": "### Classify: 'Will definitely buy again!' ### Answer: positive"},
]

TEST_INPUTS = [
    ("This product is fantastic!", "positive"),
    ("Complete waste of money.", "negative"),
    ("It is okay, nothing special.", "neutral"),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def load_model_with_lora(rank: int) -> tuple:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base, cfg)
    total, trainable = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable, total

def train_and_eval(rank: int, output_dir: str) -> dict:
    model, trainable, total = load_model_with_lora(rank)
    dataset = Dataset.from_list(TRAIN_EXAMPLES)

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=5,
        per_device_train_batch_size=2, gradient_accumulation_steps=2,
        learning_rate=2e-4, fp16=True, logging_steps=5,
        report_to="none", save_strategy="no",
    )
    trainer = SFTTrainer(model=model, train_dataset=dataset, dataset_text_field="text",
                         max_seq_length=128, tokenizer=tokenizer, args=args)
    trainer.train()

    # Evaluate
    model.eval()
    correct = 0
    for text_in, expected in TEST_INPUTS:
        prompt = f"### Classify: '{text_in}' ### Answer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if expected in pred:
            correct += 1

    return {
        "rank": rank,
        "trainable_params": trainable,
        "trainable_pct": f"{trainable/total*100:.3f}%",
        "accuracy": f"{correct}/{len(TEST_INPUTS)}",
    }

results = []
for r in [4, 16]:
    print(f"\nTraining LoRA rank={r}...")
    res = train_and_eval(r, f"./lora-r{r}")
    results.append(res)

print(f"\n{'Rank':<8} {'Trainable Params':<20} {'% of Total':<15} {'Accuracy'}")
print("-" * 55)
for r in results:
    print(f"{r['rank']:<8} {r['trainable_params']:<20,} {r['trainable_pct']:<15} {r['accuracy']}")
```

---

## Exercise 3 — Synthetic data pipeline (Stretch)

Build an end-to-end pipeline: generate synthetic training data with GPT-4o-mini, format it for SFT, audit quality, and output a training-ready JSONL file.

```python
import os
import json
import random
from openai import OpenAI
from collections import Counter

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TASK = "Classify customer support tickets into: billing, technical, account, or shipping"

def generate_batch(category: str, n: int = 5) -> list[dict]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Generate {n} realistic customer support tickets for category: {category}
Vary length (1-3 sentences) and writing style (formal, casual, frustrated).
Return JSON: {{"examples": [{{"ticket": "...", "category": "{category}"}}]}}"""
        }],
        temperature=0.9, max_tokens=800,
        response_format={"type": "json_object"}
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("examples", [])

def to_sft_format(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Classify the customer support ticket into: billing, technical, account, or shipping. Respond with only the category."},
            {"role": "user", "content": example["ticket"]},
            {"role": "assistant", "content": example["category"]}
        ]
    }

def audit(examples: list[dict]) -> None:
    categories = [ex["messages"][2]["content"] for ex in examples]
    lengths = [len(ex["messages"][1]["content"].split()) for ex in examples]
    print(f"Total examples:  {len(examples)}")
    print(f"Category balance: {dict(Counter(categories))}")
    print(f"Avg ticket length: {sum(lengths)/len(lengths):.1f} words")
    print(f"Min/max length: {min(lengths)} / {max(lengths)} words")

    # Check for empty outputs
    empties = sum(1 for ex in examples if not ex["messages"][2]["content"].strip())
    print(f"Empty outputs: {empties}")

# Generate balanced dataset
CATEGORIES = ["billing", "technical", "account", "shipping"]
all_raw = []
for cat in CATEGORIES:
    print(f"Generating {cat} examples...")
    batch = generate_batch(cat, n=8)
    all_raw.extend(batch)

# Format and audit
formatted = [to_sft_format(ex) for ex in all_raw if ex.get("ticket") and ex.get("category")]
random.shuffle(formatted)

print("\n--- Dataset Audit ---")
audit(formatted)

# Write to JSONL
output_path = "support_tickets_train.jsonl"
with open(output_path, "w") as f:
    for ex in formatted:
        f.write(json.dumps(ex) + "\n")

print(f"\nWrote {len(formatted)} examples to {output_path}")
```

---

[[05-when-to-fine-tune]] | [[07-interview-questions]]
