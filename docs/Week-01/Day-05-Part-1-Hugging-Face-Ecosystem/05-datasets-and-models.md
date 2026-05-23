# Datasets and Model Cards

The `datasets` library is the Hugging Face counterpart to `pandas` for ML datasets — it handles sharded files, streaming from disk, and Arrow-backed fast processing. Model cards document what a model does, how it was trained, and where it fails.

## Learning objectives

- Load and explore Hub datasets with `datasets`
- Stream large datasets without loading them fully into memory
- Process datasets with `.map()` for tokenization and feature extraction
- Read and write model cards using the `ModelCard` API

---

## Loading datasets

```python
from datasets import load_dataset

# Load a small benchmark dataset (downloads and caches automatically)
dataset = load_dataset("openai/gsm8k", "main")
print(dataset)
# DatasetDict({
#     train: Dataset({features: ['question', 'answer'], num_rows: 7473})
#     test:  Dataset({features: ['question', 'answer'], num_rows: 1319})
# })

# Access splits
train = dataset["train"]
test = dataset["test"]

# Inspect examples
print(train[0])
# {'question': 'Natalia sold clips...', 'answer': '72'}

# Filter, slice, and select
hard_examples = train.filter(lambda x: len(x["answer"]) > 3)
first_100 = train.select(range(100))
```

---

## Exploring dataset schema and statistics

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("squad", split="validation[:500]")

# Schema
print(dataset.features)
# {'id': Value(dtype='string'), 'title': Value(dtype='string'),
#  'context': Value(dtype='string'), 'question': Value(dtype='string'),
#  'answers': Sequence({'text': [...], 'answer_start': [...]})}

# Convert to pandas for quick exploration
df = dataset.to_pandas()
print(df.describe(include="all"))
print(f"\nAvg context length: {df['context'].str.len().mean():.0f} chars")
print(f"Avg question length: {df['question'].str.len().mean():.0f} chars")
print(f"Unique titles: {df['title'].nunique()}")
```

---

## Streaming large datasets

Datasets like The Pile, RedPajama, or Common Crawl are hundreds of GB. Stream them instead of downloading.

```python
from datasets import load_dataset

# streaming=True: data is loaded in chunks, never fully in memory
streamed = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True
)

# Iterate over batches
for i, batch in enumerate(streamed.iter(batch_size=32)):
    texts = batch["text"]
    print(f"Batch {i}: {len(texts)} examples, first: {texts[0][:80]}...")
    if i >= 2:
        break

# Take a fixed number of examples for quick experiments
sample = list(streamed.take(1000))
print(f"Sampled {len(sample)} examples")
```

---

## Processing datasets with .map()

`.map()` applies a function to every example (or batch) and caches the result.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dataset = load_dataset("imdb", split="train[:1000]")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors=None  # Return lists, not tensors (for dataset caching)
    )

# Tokenize in batches — much faster than row-by-row
tokenized = dataset.map(
    tokenize,
    batched=True,
    batch_size=64,
    num_proc=2,              # Parallel processing
    remove_columns=["text"]  # Drop original text column
)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print(tokenized)
# Dataset with input_ids, attention_mask, label — ready for PyTorch DataLoader
```

---

## Pushing a dataset to the Hub

```python
from datasets import Dataset, DatasetDict

# Create a dataset from Python objects
examples = [
    {"question": "What is RAG?", "answer": "Retrieval-Augmented Generation", "category": "rag"},
    {"question": "What is HNSW?", "answer": "Hierarchical Navigable Small World", "category": "vector_db"},
    {"question": "What is LoRA?", "answer": "Low-Rank Adaptation", "category": "fine_tuning"},
]

train_ds = Dataset.from_list(examples)
eval_ds = Dataset.from_list(examples[:1])

full_dataset = DatasetDict({"train": train_ds, "validation": eval_ds})

# Push to Hub (creates or updates the repository)
full_dataset.push_to_hub(
    "your-username/llm-course-qa",
    token=os.getenv("HF_TOKEN"),
    private=True
)
```

---

## Reading and writing model cards

Model cards document intended use, training data, evaluation results, and limitations. The `huggingface_hub` SDK includes a `ModelCard` class.

```python
from huggingface_hub import ModelCard, ModelCardData

# Read a model card
card = ModelCard.load("mistralai/Mistral-7B-Instruct-v0.3")
print(card.data)           # Parsed YAML frontmatter
print(card.text[:500])     # Raw markdown content

# Create a new model card
card_data = ModelCardData(
    language=["en"],
    license="apache-2.0",
    library_name="transformers",
    tags=["text-generation", "llm", "fine-tuned"],
    datasets=["squad"],
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    pipeline_tag="text-generation",
    metrics=[
        {"name": "Faithfulness (RAGAS)", "type": "faithfulness", "value": 0.87},
        {"name": "Answer Relevancy", "type": "answer_relevancy", "value": 0.91},
    ]
)

card_content = f"""---
{card_data.to_yaml()}
---

# My Fine-Tuned Q&A Model

Fine-tuned from Llama 3.1 8B Instruct on a domain-specific Q&A dataset.

## Intended use

Customer support Q&A for software products. Do not use for medical, legal, or financial advice.

## Evaluation results

| Metric | Score |
|--------|-------|
| Faithfulness (RAGAS) | 0.87 |
| Answer Relevancy | 0.91 |
| Context Recall | 0.83 |

## Limitations

- English only
- Domain-specific: may perform poorly on general knowledge questions
- Context window: 4096 tokens

## Training details

- Base model: Llama 3.1 8B Instruct
- Method: QLoRA (r=16, alpha=32, dropout=0.05)
- Training data: 5,000 domain-specific QA pairs
- Hardware: 1x A100 40GB, 2 hours
"""

new_card = ModelCard(card_content)
# new_card.push_to_hub("your-username/my-model")
print("Card created successfully")
```

> [!success] Model cards are infrastructure
> Treat your model card as code. It's version-controlled, it communicates expected behavior to users, and it provides the paper trail needed for compliance audits. A model without a card is a black box — even to yourself 6 months later.

---

[[04-spaces]] | [[06-practice-exercises]]
