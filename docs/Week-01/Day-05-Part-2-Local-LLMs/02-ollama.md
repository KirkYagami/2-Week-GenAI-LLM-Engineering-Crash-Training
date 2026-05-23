# Ollama

Ollama is the fastest path to running LLMs locally. Install it, pull a model, and you have a local OpenAI-compatible API endpoint in under five minutes. It handles GGUF model downloading, GPU detection, context management, and REST API serving — all automatically.

## Learning objectives

- Install Ollama and pull models from the library
- Use the Ollama REST API and Python `ollama` client
- Drop in Ollama as an OpenAI API replacement using `base_url`
- Configure model parameters and run models efficiently

---

## Installation and first run

```bash
# macOS
brew install ollama
# or: curl -fsSL https://ollama.com/install.sh | sh

# Windows (download installer from https://ollama.com)
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server
ollama serve   # runs on http://localhost:11434

# In a new terminal: pull and run a model
ollama pull llama3.2:3b          # 2.0 GB — fast to download, runs on 8GB RAM
ollama pull llama3.1:8b          # 4.7 GB — best 8B model available
ollama pull mistral:7b           # 4.1 GB — fast, good for coding
ollama pull nomic-embed-text     # 274 MB — embeddings

# Interactive chat
ollama run llama3.1:8b "Explain attention mechanisms in 3 sentences."
```

---

## Python client

```python
import ollama

# Simple generation
response = ollama.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "What is the difference between RAG and fine-tuning?"}
    ]
)
print(response["message"]["content"])
print(f"\nTokens: {response['eval_count']} tokens, {response['eval_duration']/1e9:.1f}s")

# Streaming
for chunk in ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Write a haiku about vector databases."}],
    stream=True
):
    print(chunk["message"]["content"], end="", flush=True)
print()
```

---

## OpenAI-compatible API

Ollama exposes `/v1/chat/completions` — the same endpoint as OpenAI. You can use the OpenAI Python client with `base_url` pointing to your local Ollama.

```python
from openai import OpenAI

# Point to local Ollama — works with any OpenAI SDK code
local_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Any non-empty string — Ollama ignores it
)

response = local_client.chat.completions.create(
    model="llama3.1:8b",
    messages=[
        {"role": "system", "content": "You answer questions about machine learning."},
        {"role": "user", "content": "Explain cosine similarity in 2 sentences."}
    ],
    temperature=0.0,
    max_tokens=200
)
print(response.choices[0].message.content)
```

This means any code written for OpenAI works with local Ollama by changing two lines. Useful for:
- Testing locally before paying for API calls
- Switching between local (dev) and API (prod) via environment variable
- Cost-free iteration on prompts and pipelines

```python
import os
from openai import OpenAI

# Switch between local and API via environment variable
USE_LOCAL = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

if USE_LOCAL:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    model = "llama3.1:8b"
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "gpt-4o-mini"

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello, what can you do?"}],
    max_tokens=100
)
print(f"[{model}] {response.choices[0].message.content}")
```

---

## Embeddings with Ollama

```python
import ollama
import numpy as np

def embed_local(texts: list[str], model: str = "nomic-embed-text") -> np.ndarray:
    """Embed texts using a local Ollama model."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        embeddings.append(response["embedding"])
    return np.array(embeddings)

# Usage
texts = [
    "How does vector search work?",
    "What is approximate nearest neighbor?",
    "The stock market rose 2% today.",
]

embs = embed_local(texts)
embs_normalized = embs / np.linalg.norm(embs, axis=1, keepdims=True)
sim = embs_normalized @ embs_normalized.T

print(f"Dimension: {embs.shape[1]}")
print(f"Q0 ↔ Q1 similarity: {sim[0,1]:.3f}")   # Should be high (both about vector search)
print(f"Q0 ↔ Q2 similarity: {sim[0,2]:.3f}")   # Should be low (off-topic)
```

---

## Modelfile: customizing model behavior

Ollama's Modelfile lets you create custom model variants with preset parameters and system prompts.

```dockerfile
# Modelfile — save as ./Modelfile
FROM llama3.1:8b

# Set the temperature and context window
PARAMETER temperature 0.0
PARAMETER num_ctx 8192
PARAMETER top_k 40
PARAMETER top_p 0.9

# Custom system prompt baked in
SYSTEM """You are a RAG evaluation assistant. Your job is to analyze whether
LLM answers are faithful to the provided context.

Rules:
1. Score faithfulness from 0.0 to 1.0.
2. List any claims not supported by context.
3. Never add information from outside the context.
4. Return structured JSON."""
```

```bash
# Build and test the custom model
ollama create rag-evaluator -f ./Modelfile
ollama run rag-evaluator "Evaluate: Context: 'Python was released in 1991.' Answer: 'Python was released in 1991 by a team at MIT.'"
```

```python
# Use the custom model in Python
response = ollama.chat(
    model="rag-evaluator",
    messages=[{
        "role": "user",
        "content": "Context: 'The refund policy allows returns within 14 days.'\nAnswer: 'Returns are accepted within 14 days of purchase.'"
    }]
)
print(response["message"]["content"])
```

---

## Model management

```bash
# List downloaded models and their sizes
ollama list

# Remove a model to free disk space
ollama rm llama3.2:3b

# Show model metadata
ollama show llama3.1:8b

# Pull a specific quantization version
ollama pull llama3.1:8b-instruct-q4_K_M   # Q4 K-mean Medium — recommended
ollama pull llama3.1:8b-instruct-q8_0     # Q8 — higher quality, 2× larger
ollama pull llama3.1:8b-instruct-fp16     # Full precision — requires 16+ GB VRAM
```

> [!tip] Recommended quantizations
> For most use cases: `q4_K_M` — best balance of quality and size. It uses K-means quantization for better accuracy than simple `q4_0`. Only step up to `q8_0` if you're doing tasks where every percentage point of accuracy matters and have the VRAM.

---

[[01-why-run-locally]] | [[03-llama-cpp]]
