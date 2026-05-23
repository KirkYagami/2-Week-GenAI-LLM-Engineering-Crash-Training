# Quantization

Quantization compresses model weights from high-precision floats to lower-precision integers. The goal is the same as image compression: reduce file size with minimal perceptible quality loss. Understanding quantization helps you choose the right format for your hardware and quality requirements.

## Learning objectives

- Understand the difference between GGUF quantization levels
- Know when to use Q4, Q6, and Q8 formats
- Apply GPTQ and AWQ quantization for GPU inference
- Measure quality degradation across quantization levels

---

## What quantization does

A model weight is originally stored as a float32 (4 bytes, 32 bits). Quantization maps these values to integers with fewer bits.

```
float32 (32 bits):  0.15234375  → exact value
float16 (16 bits):  0.152       → 2× smaller, tiny loss
int8    (8 bits):   24          → 4× smaller, small loss
int4    (4 bits):   1           → 8× smaller, noticeable loss
int2    (2 bits):   0           → 16× smaller, significant loss
```

The quantization error accumulates across hundreds of layers. At 4-bit, the error is usually small enough that most users can't tell the difference in conversation quality. At 2-bit, quality degrades substantially.

---

## GGUF quantization types

llama.cpp and Ollama use GGUF quantization with several variants. The naming follows a pattern:

| Format | Bits | Size (7B model) | Quality | Use case |
|--------|------|-----------------|---------|----------|
| F32 | 32 | 28 GB | Reference | Never — too large |
| F16 | 16 | 14 GB | ≈100% | GPU training/inference |
| Q8_0 | 8 | 7.2 GB | ~99.5% | High quality, have VRAM |
| Q6_K | 6 | 5.5 GB | ~99% | Good quality, less VRAM |
| Q5_K_M | 5 | 4.8 GB | ~98.5% | Balanced |
| **Q4_K_M** | **4** | **4.1 GB** | **~97%** | **Best default choice** |
| Q4_0 | 4 | 3.9 GB | ~96% | Slightly worse than Q4_K_M |
| Q3_K_M | 3 | 3.1 GB | ~94% | Size-constrained only |
| Q2_K | 2 | 2.2 GB | ~88% | Avoid unless necessary |

**K-quantization (K_M, K_S, K_L):** Uses k-means clustering to find optimal quantization centroids. Better quality than simple quantization at the same bit width. `K_M` (medium) is the standard choice; `K_S` is smaller (fewer bits for non-critical layers); `K_L` is larger.

```python
def recommend_quantization(vram_gb: float, quality_priority: str = "balanced") -> str:
    """Recommend GGUF quantization given available VRAM and quality need."""

    if quality_priority == "max_quality":
        if vram_gb >= 14:
            return "F16 — maximum quality, full float16"
        if vram_gb >= 7:
            return "Q8_0 — near-lossless, ~99.5% of F16 quality"
        if vram_gb >= 5.5:
            return "Q6_K — very high quality, ~99%"
    elif quality_priority == "balanced":
        if vram_gb >= 8:
            return "Q5_K_M — excellent balance of size and quality"
        if vram_gb >= 4.5:
            return "Q4_K_M — best default; ~97% quality, 4.1GB for 7B"
        if vram_gb >= 3.5:
            return "Q4_0 — slightly less accurate than Q4_K_M"
    else:  # max_compression
        if vram_gb >= 3.5:
            return "Q3_K_M — compact, acceptable quality"
        return "Q2_K — last resort; noticeable quality loss"

    return "Q4_K_M — default recommendation"

print(recommend_quantization(6.0))  # Q4_K_M or Q5_K_M
print(recommend_quantization(16.0, "max_quality"))  # Q8_0
print(recommend_quantization(2.5, "max_compression"))  # Q2_K
```

---

## GPTQ — GPU-optimized quantization

GPTQ uses the Hessian of the loss to find optimal quantization weights, losing much less quality than naive rounding. It targets GPU inference (not CPU like GGUF).

```python
# Load a pre-quantized GPTQ model from the Hub
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Many models have pre-quantized GPTQ versions uploaded by TheBloke
# Example: "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",
    tokenizer=AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
)

# Load pre-quantized model (no quantization happens at load time)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    device_map="auto",
    quantization_config=gptq_config
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
print(f"Loaded GPTQ model on: {model.device}")
```

---

## AWQ — Activation-aware Weight Quantization

AWQ (2023) identifies which weights are most important by examining activations, then quantizes unimportant weights more aggressively. It achieves better quality than GPTQ at 4-bit.

```python
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load an AWQ-quantized model
model_id = "casperhansen/mistral-7b-instruct-v0.1-awq"
model = AutoAWQForCausalLM.from_quantized(
    model_id,
    fuse_layers=True,     # Fuse attention layers for faster inference
    trust_remote_code=False,
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

# Generate
inputs = tokenizer("Explain cosine similarity:", return_tensors="pt").to(model.device)
with __import__("torch").no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Measuring quality degradation

Compare quantized vs full-precision outputs to understand the real quality gap.

```python
import os
from openai import OpenAI

# Use Ollama to compare quantization levels side-by-side
client_q4 = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

EVAL_PROMPTS = [
    "What is 237 × 48? Show your working.",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "Write a Python function that checks if a string is a palindrome.",
    "The user asked: 'How do I learn machine learning?' Give a structured 5-step plan.",
]

def eval_model(model: str, prompts: list[str]) -> list[str]:
    responses = []
    for prompt in prompts:
        resp = client_q4.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        responses.append(resp.choices[0].message.content)
    return responses

# Compare Q4 vs Q8 (if you have both pulled in Ollama)
models_to_test = [
    "llama3.1:8b",          # Default Q4_K_M in Ollama
    "llama3.1:8b-instruct-q8_0",  # Q8 version
]

for model in models_to_test:
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    responses = eval_model(model, EVAL_PROMPTS[:1])  # Test one prompt
    print(f"Response: {responses[0][:200]}...")
```

> [!success] Practical quantization guidance
> For most production use cases: **Q4_K_M is sufficient**. Independent benchmarks consistently show < 3% quality degradation on MMLU, HellaSwag, and GSM8k compared to F16. The exceptions are: complex multi-step math (where Q8+ is noticeably better), and code generation with edge cases. If your use case involves either, step up to Q6_K or Q8_0 if you have the VRAM.

---

| Quantization | When to use |
|-------------|-------------|
| F16 | Fine-tuning, quality-critical production with plenty of VRAM |
| Q8_0 | High-quality production where VRAM allows |
| Q5_K_M | Best balance for 8–12 GB VRAM systems |
| **Q4_K_M** | **Default choice for most use cases** |
| Q3_K_M | Only when fitting the model is the priority |
| Q2_K | Avoid — quality loss is usually unacceptable |

---

[[03-llama-cpp]] | [[05-when-to-run-locally]]
