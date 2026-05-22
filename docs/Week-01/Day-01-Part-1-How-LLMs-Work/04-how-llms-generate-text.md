# How LLMs Generate Text

LLMs don't "think of" the next word. They compute a probability distribution over the entire vocabulary and then **sample** from it. The sampling strategy you choose shapes everything from creativity to factual accuracy.

## Learning objectives

- Explain autoregressive generation and why it's sequential
- Compare greedy, temperature, top-k, and top-p sampling
- Set `temperature`, `top_p`, and `max_tokens` correctly for a given task
- Explain why reasoning models (o3, o4-mini) behave differently

---

## Autoregressive generation

LLMs generate text one token at a time, left to right. Each new token is conditioned on all previous tokens:

```
P(token_5 | token_1, token_2, token_3, token_4)
```

The model cannot "go back" and change an earlier token. This is why getting the model to reason *before* answering — via chain-of-thought — improves output quality: it commits earlier tokens to a reasoning path that constrains and improves later tokens.

```python
import openai

client = openai.OpenAI()

# Each call to the API generates one "response" which internally
# generates tokens one at a time via autoregressive decoding
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=5,
    temperature=0,  # greedy — always picks the highest-probability token
)
print(response.choices[0].message.content)
# Output: " Paris." (deterministic with temperature=0)
```

---

## The logits-to-token pipeline

After the transformer's final layer, the model produces a **logit** vector — one raw score per vocabulary token (~200,000 scores for GPT-4o):

```
[final transformer layer output]
         ↓
[Linear projection] → 200,000 raw logit scores
         ↓
[Apply temperature] → scale logits
         ↓
[Apply top-k / top-p filtering] → zero out low-probability tokens
         ↓
[Softmax] → probability distribution
         ↓
[Sample] → one token index
         ↓
[Decode] → "Paris"
```

---

## Greedy decoding

Always pick the highest-probability token. Deterministic but boring.

```python
import torch
import torch.nn.functional as F

def greedy_decode(logits: torch.Tensor) -> int:
    return logits.argmax().item()

# The problem with greedy: it gets stuck in repetitive loops
# "The cat sat on the mat. The cat sat on the mat. The cat..."
# Use temperature=0 in APIs for near-greedy behavior
```

**When to use:** Extraction tasks, structured output, or when you need determinism in tests.

---

## Temperature

Temperature scales the logits before softmax, controlling how peaked or flat the distribution is.

```python
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        # Effectively greedy — return a one-hot distribution
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    return F.softmax(logits / temperature, dim=-1)

# Demonstration
logits = torch.tensor([3.0, 1.5, 0.5, 0.2])  # 4-token vocab

for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
    probs = apply_temperature(logits, temp)
    print(f"T={temp}: {probs.tolist()}")

# T=0.1:  [0.98, 0.02, 0.00, 0.00]  ← very peaked, near-deterministic
# T=1.0:  [0.67, 0.22, 0.08, 0.03]  ← original distribution
# T=2.0:  [0.40, 0.28, 0.19, 0.14]  ← much flatter, more random
```

**Temperature guide by task:**

| Task | Recommended Temperature |
|------|------------------------|
| Fact extraction, classification | 0.0 – 0.2 |
| Summarization, Q&A | 0.3 – 0.5 |
| Code generation | 0.2 – 0.4 |
| Writing, brainstorming | 0.7 – 1.0 |
| Creative fiction, poetry | 1.0 – 1.2 |
| Avoid anything above 1.5 | Output degrades rapidly |

---

## Top-k sampling

Restrict sampling to only the top `k` tokens. Prevents the model from ever picking a very improbable token.

```python
def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top k."""
    top_k_vals, _ = torch.topk(logits, k)
    threshold = top_k_vals[-1]
    filtered = logits.clone()
    filtered[filtered < threshold] = float("-inf")
    return F.softmax(filtered, dim=-1)

logits = torch.tensor([3.0, 1.5, 0.5, 0.2, -0.5, -2.0])
probs_k10 = top_k_filter(logits, k=3)
print(probs_k10)  # Only top 3 tokens have non-zero probability
```

**Problem with top-k:** `k=50` is too wide when the distribution is sharp (one obvious token), and too narrow when the distribution is flat (many reasonable tokens).

---

## Top-p (nucleus) sampling

Instead of a fixed count `k`, take the smallest set of tokens whose cumulative probability exceeds `p`. Adapts to the distribution shape.

```python
def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep only the top tokens whose cumulative probability >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens above the threshold (but keep the first one that crosses)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return F.softmax(filtered_logits, dim=-1)

# When the model is very confident: top-p=0.9 picks just 1-2 tokens
# When the model is uncertain: top-p=0.9 picks 10-20 tokens
# This adaptive behavior is why top-p outperforms top-k in practice
```

**In practice:** Set `top_p=1.0` (effectively disabled) and use only temperature, OR set `temperature=1.0` and use only top-p. Combining both often causes unexpected interactions.

---

## Using sampling parameters in the API

```python
import openai, anthropic

openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic()

# Creative writing — high temperature
creative_response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write the opening line of a noir novel."}],
    temperature=1.0,
    max_tokens=100,
    # top_p=1.0,  # leave at default when using temperature
)

# Data extraction — near-greedy
extraction_response = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    temperature=0.0,  # deterministic
    messages=[{
        "role": "user",
        "content": 'Extract the invoice number from: "Invoice #INV-2024-8821 dated March 15"',
    }],
)

print("Creative:", creative_response.choices[0].message.content)
print("Extraction:", extraction_response.content[0].text)
```

---

## Stop sequences

Stop sequences tell the model to stop generating when it produces a specific string. Crucial for structured output and preventing runaway responses.

```python
# Stop before the model starts a new "User:" turn in simulated dialogue
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Simulate a support conversation."}],
    stop=["User:", "Human:", "\n\n---"],
    max_tokens=500,
)

# Stop after JSON object closes
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": 'Return JSON: {"name": "Alice", "age":'}],
    stop=["}"],
    max_tokens=20,
)
# Output will be: ' 30' — the model stops before producing the closing brace
```

> [!warning] max_tokens is not optional
> Always set `max_tokens`. Without it, the model can generate until the context window is full — costing you money and delivering a garbage response. For most tasks: 256–2048. For summaries of long docs: 1024–4096.

---

## Reasoning models: a different paradigm

OpenAI's o-series (o1, o3, o4-mini) and Anthropic's extended thinking mode work differently. They generate a hidden chain-of-thought before producing the visible answer.

```python
# o4-mini uses a "reasoning effort" parameter instead of temperature
response = openai_client.chat.completions.create(
    model="o4-mini",
    messages=[{"role": "user", "content": "What is 17 × 23 + sqrt(289)?"}],
    reasoning_effort="high",  # "low", "medium", or "high"
    max_completion_tokens=2000,  # includes reasoning tokens
)
# Note: temperature and top_p are NOT used with o-series models
print(response.choices[0].message.content)

# Anthropic extended thinking
response = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000,  # how many tokens the model can spend reasoning
    },
    messages=[{"role": "user", "content": "Solve: If a snail crawls 3cm/hour..."}],
)
for block in response.content:
    if block.type == "thinking":
        print("REASONING:", block.thinking[:200], "...")
    else:
        print("ANSWER:", block.text)
```

> [!tip] When to use reasoning models
> Use o3/o4-mini or extended thinking for: multi-step math, complex code generation, tasks where being wrong is expensive. Don't use them for: simple Q&A, quick classification, tasks that need low latency. They're 5–20× slower and more expensive.

---

## Beam search (not used in modern LLM APIs)

Beam search maintains the top `B` candidate sequences simultaneously. It was standard in translation models but is **not used** in GPT-4o, Claude, or other chat models:

- Produces repetitive, generic output
- Computationally expensive at scale
- Incompatible with streaming

Beam search still appears in specialized models (like Whisper for speech-to-text). For chat and generation, sampling always wins.

---

> [!success] Key takeaway
> For extraction and classification: `temperature=0`. For generation and creativity: `temperature=0.7–1.0`. Always set `max_tokens`. Use reasoning models when accuracy matters more than speed.

[[03-context-windows]] | [[05-practice-exercises]]
