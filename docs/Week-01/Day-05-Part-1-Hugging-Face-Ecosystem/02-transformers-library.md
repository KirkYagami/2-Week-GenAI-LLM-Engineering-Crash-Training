# Transformers Library

The `transformers` library by Hugging Face is the standard interface for loading and running pre-trained models. It abstracts away the differences between architectures — the same pipeline API works whether you're running a 70M-parameter classifier or a 70B-parameter instruct model.

## Learning objectives

- Run inference with `pipeline()` for common NLP tasks
- Understand the tokenizer → model → decode workflow
- Load models with quantization for memory-constrained hardware
- Apply chat templates for instruction-tuned models

---

## The pipeline API

`pipeline()` is the fastest path from model name to inference. It handles tokenization, inference, and decoding automatically.

```python
from transformers import pipeline
import torch

# Sentiment analysis — zero setup
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

results = classifier([
    "This course is excellent — best investment I've made.",
    "The content is too dense and hard to follow.",
])

for text, result in zip(["positive", "negative"], results):
    print(f"Label: {result['label']}, Score: {result['score']:.3f}")
# Label: POSITIVE, Score: 0.999
# Label: NEGATIVE, Score: 0.994
```

```python
# Named Entity Recognition
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
    device=-1
)

text = "Sam Altman, CEO of OpenAI, spoke at the San Francisco conference."
entities = ner(text)
for ent in entities:
    print(f"[{ent['entity_group']:4}] {ent['word']:20} ({ent['score']:.2f})")
# [PER ] Sam Altman           (0.99)
# [ORG ] OpenAI               (0.99)
# [LOC ] San Francisco        (0.98)
```

```python
# Text generation — instruct model
generator = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",  # 3.8B, fits on 6GB VRAM
    torch_dtype=torch.bfloat16,
    device_map="auto",  # auto-distributes across available GPUs/CPU
    trust_remote_code=True
)

output = generator(
    "Explain the difference between RAG and fine-tuning in one paragraph.",
    max_new_tokens=200,
    do_sample=False,
    temperature=1.0   # ignored when do_sample=False
)
print(output[0]["generated_text"])
```

---

## Tokenizers

The tokenizer converts text to token IDs and back. Understanding it prevents subtle bugs.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

text = "The quick brown fox jumps over the lazy dog."

# Encode to token IDs
encoded = tokenizer(text, return_tensors="pt")
print(f"Input IDs: {encoded['input_ids']}")
print(f"Token count: {encoded['input_ids'].shape[1]}")

# Decode back to text
decoded = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
print(f"Decoded: {decoded}")

# Inspect individual tokens
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"\nTokens: {tokens}")
print(f"IDs:    {ids}")
# Tokens: ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', ...]
# (Ġ = space prefix for GPT-style byte-pair encoding)

# Count tokens in a string without full encoding
def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

print(f"\nToken count: {count_tokens(text, tokenizer)}")
```

---

## Chat templates for instruction models

Instruction-tuned models expect input in a specific format — different for each model family. Chat templates handle this automatically.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to compute fibonacci numbers."},
]

# apply_chat_template converts messages to the model's expected format
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(prompt)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful coding assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Write a Python function to compute fibonacci numbers.<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
```

> [!warning] Always use apply_chat_template
> Manually formatting prompts with `[INST]` tags or `<|im_start|>` is error-prone and model-version-specific. `apply_chat_template` reads the template from the tokenizer config — use it for every instruction model to avoid silent formatting bugs that reduce quality without raising errors.

---

## Loading models: AutoModel vs AutoModelForCausalLM

```python
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel
)
import torch

# For text generation
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.bfloat16,  # Use bfloat16 to halve memory vs float32
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Efficient generation with no_grad
def generate(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode only newly generated tokens (skip input prompt)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# For embeddings
embed_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
embed_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def get_embedding(text: str) -> list[float]:
    inputs = embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Mean pool over token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.tolist()
```

---

## 4-bit quantization with BitsAndBytes

Loading a 7B model in full float32 requires ~28 GB of VRAM. Quantization reduces this dramatically.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 4-bit NormalFloat quantization (QLoRA method)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Mistral 7B: ~14 GB float16 → ~4 GB in 4-bit
# This fits on a free-tier Colab T4 GPU (16 GB)
```

| Model size | Float32 | Float16 | Int8 | Int4 |
|-----------|---------|---------|------|------|
| 7B params | 28 GB | 14 GB | 7 GB | 3.5 GB |
| 13B params | 52 GB | 26 GB | 13 GB | 6.5 GB |
| 70B params | 280 GB | 140 GB | 70 GB | 35 GB |

> [!tip] Free GPU options for experimenting
> Google Colab T4 (free): 16 GB VRAM — runs 7B models in 4-bit.
> Kaggle P100 (free): 16 GB VRAM — same.
> Hugging Face ZeroGPU (Spaces free tier): shared A100, up to 40 GB.
> For development and testing, these are sufficient for Mistral-7B and Llama-3.1-8B.

---

## Text classification pipeline with batching

```python
from transformers import pipeline

# Zero-shot classification — no fine-tuning needed
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

candidate_labels = ["billing", "technical support", "account management", "general inquiry"]

texts = [
    "I was charged twice for last month's subscription.",
    "My app keeps crashing on iOS 17.",
    "How do I change my email address?",
    "What time does customer service open?",
]

# Batch classify for efficiency
results = classifier(texts, candidate_labels, multi_label=False)

for text, result in zip(texts, results):
    top = result["labels"][0]
    score = result["scores"][0]
    print(f"[{top:20} {score:.2f}] {text[:50]}...")
```

---

[[01-hugging-face-hub]] | [[03-inference-api]]
