# Inference API

The Hugging Face Inference API lets you run models hosted on the Hub without managing any GPU infrastructure. You send a POST request with your text; you get back the model's output. For prototyping and low-volume production use, it's faster to set up than a self-hosted GPU server.

## Learning objectives

- Call the Inference API using the `InferenceClient` Python SDK
- Handle rate limits and retries correctly
- Use the serverless API for tasks: text generation, embeddings, classification, image generation
- Understand when to use Inference Endpoints vs the serverless API

---

## InferenceClient — the Python SDK

The `huggingface_hub` package includes `InferenceClient`, which handles authentication, retries, and response parsing.

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Text generation
response = client.text_generation(
    prompt="Explain gradient descent in one paragraph, as if to a software engineer.",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
)
print(response)
```

---

## Chat completions (OpenAI-compatible endpoint)

Many models on the Hub expose an OpenAI-compatible `/v1/chat/completions` endpoint. This means you can use the OpenAI Python client with a custom `base_url`.

```python
import os
from openai import OpenAI

# Use HF Inference API with the OpenAI client
hf_client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=os.getenv("HF_TOKEN"),
)

response = hf_client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "What is the difference between LSTM and Transformer?"}
    ],
    max_tokens=300,
    temperature=0.5,
)
print(response.choices[0].message.content)
```

> [!info] OpenAI-compatible models on HF
> Any model with "Inference Endpoints" enabled supports the `/v1/chat/completions` route. This lets you swap between OpenAI and open-source models by changing `base_url` and `model` — no other code changes needed.

---

## Embeddings via Inference API

```python
import os
import numpy as np
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

def embed_texts(texts: list[str], model: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Batch embed texts using the Inference API."""
    embeddings = client.feature_extraction(
        text=texts,
        model=model,
        normalize=True
    )
    return np.array(embeddings)

# Semantic similarity
texts = [
    "How do I cancel my subscription?",
    "What is the process to unsubscribe?",
    "Tell me about your pricing plans.",
]

embs = embed_texts(texts)

# Cosine similarity (vectors already normalized)
similarity_matrix = embs @ embs.T
print("Similarity matrix:")
for i, text_i in enumerate(texts):
    for j, text_j in enumerate(texts):
        if i < j:
            print(f"  {i}↔{j}: {similarity_matrix[i,j]:.3f}  |  '{text_i[:30]}...' ↔ '{text_j[:30]}...'")
```

---

## Classification and NER

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Sentiment analysis
sentiment = client.text_classification(
    text="The new update broke all my workflows. Very frustrated.",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print(f"Sentiment: {sentiment[0]['label']} ({sentiment[0]['score']:.3f})")

# Zero-shot classification (no fine-tuning needed)
result = client.zero_shot_classification(
    text="I need to update my payment method.",
    labels=["billing", "account", "technical support", "general"],
    model="facebook/bart-large-mnli"
)
# Sort by score
ranked = sorted(zip(result["labels"], result["scores"]), key=lambda x: -x[1])
for label, score in ranked:
    print(f"  {label:<20} {score:.3f}")

# Named entity recognition
entities = client.token_classification(
    text="Elon Musk founded SpaceX in Hawthorne, California in 2002.",
    model="dslim/bert-base-NER"
)
for entity in entities:
    if entity["entity_group"] in ("PER", "ORG", "LOC"):
        print(f"  [{entity['entity_group']}] {entity['word']}")
```

---

## Streaming text generation

```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Streaming output — tokens arrive as they're generated
print("Streaming response:")
for token in client.text_generation(
    prompt="List 5 best practices for writing RAG prompts:",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=300,
    stream=True
):
    print(token, end="", flush=True)
print()
```

---

## Rate limits and error handling

The free Inference API has rate limits: ~300 requests/hour for most models.

```python
import os
import time
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
import logging

logger = logging.getLogger(__name__)
client = InferenceClient(token=os.getenv("HF_TOKEN"))

def call_with_retry(
    func,
    *args,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    **kwargs
):
    """Call a function with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except HfHubHTTPError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = backoff_base ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "503" in str(e) or "loading" in str(e).lower():
                # Model is loading (cold start) — wait longer
                wait = 20.0
                logger.info(f"Model loading. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")

# Usage
response = call_with_retry(
    client.text_generation,
    prompt="Summarize the key concepts of attention mechanisms.",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=150
)
print(response)
```

> [!warning] Cold start latency
> Free-tier models spin down when idle. The first request after idle may return a 503 with `"loading"` in the error message. This is normal — retry after 20–30 seconds. Paid Inference Endpoints stay warm and have no cold start.

---

## Inference API vs Inference Endpoints

| Feature | Serverless Inference API | Inference Endpoints |
|---------|------------------------|---------------------|
| Setup | None (use immediately) | Deploy via UI or API |
| Cost | Free tier + pay-per-call | Hourly instance cost |
| Cold start | Yes (up to 60s) | No (always warm) |
| Custom models | Public Hub models only | Any Hub model or private |
| Hardware | Shared GPU pool | Dedicated GPU |
| Rate limits | ~300/hour free | None (you own the instance) |
| SLA | Best effort | Uptime SLA available |

**Use serverless for:** Prototyping, development, low-volume applications, testing model quality.
**Use Endpoints for:** Production applications, latency-sensitive use cases, private models, consistent throughput.

```python
# Deploying a dedicated Inference Endpoint programmatically
from huggingface_hub import create_inference_endpoint

# endpoint = create_inference_endpoint(
#     name="my-production-endpoint",
#     repository="mistralai/Mistral-7B-Instruct-v0.3",
#     framework="pytorch",
#     task="text-generation",
#     accelerator="gpu",
#     instance_size="x1",
#     instance_type="nvidia-l4",
#     region="us-east-1",
#     type="protected",  # "public", "protected", or "private"
# )
# endpoint.wait()  # Wait for deployment
# print(f"Endpoint URL: {endpoint.url}")
```

---

[[02-transformers-library]] | [[04-spaces]]
