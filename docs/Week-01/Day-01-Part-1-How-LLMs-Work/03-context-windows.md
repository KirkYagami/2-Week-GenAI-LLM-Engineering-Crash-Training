# Context Windows

The context window is the model's working memory — everything it can "see" at once. Get this wrong and your RAG pipeline hallucination rate spikes. Get it right and you can process entire codebases or legal documents in a single call.

## Learning objectives

- Define context window and explain what goes inside it
- Compare current model context limits and their practical implications
- Identify the "lost in the middle" problem and when it matters
- Design prompts and retrieval pipelines that work within context limits

---

## What is a context window?

The context window is the maximum number of tokens a model can process in a single forward pass — input tokens + output tokens combined.

```
┌──────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW                    │
│                                                      │
│  [System Prompt] [Chat History] [Retrieved Docs]     │
│  [User Message]  ←── input tokens ──────────────►   │
│                                                      │
│  [Model Response] ←── output tokens ───────────►     │
└──────────────────────────────────────────────────────┘
```

Everything in the window is "visible" to every attention head in every layer. Nothing outside the window exists for the model. It has no implicit memory between API calls.

---

## 2025 context window landscape

| Model | Context Window | Max Output | Notes |
|-------|---------------|------------|-------|
| GPT-4o | 128,000 tokens | 16,384 | ~100 pages of text |
| GPT-4o-mini | 128,000 tokens | 16,384 | Same window, lower cost |
| o3 | 200,000 tokens | 100,000 | Chain-of-thought output can be large |
| o4-mini | 200,000 tokens | 100,000 | |
| Claude Sonnet 4.6 | 1,000,000 tokens | 64,000 | ~750 pages of text |
| Claude Opus 4.7 | 1,000,000 tokens | 64,000 | |
| Gemini 3 Pro | 1,000,000 tokens | 65,000 | |
| Llama 3.3 70B | 128,000 tokens | 8,192 | Open-weight |
| Mistral Large | 128,000 tokens | — | |

*As of May 2026. The context window race has largely settled — 1M tokens is the current ceiling for frontier models.*

> [!info] Tokens → pages rough conversion
> 1,000 tokens ≈ 750 words ≈ 1.5 pages of English text. So:
> - 128K tokens ≈ 100 pages
> - 1M tokens ≈ 750 pages (a thick novel, or a large codebase)

---

## What goes inside the context window

In a typical RAG pipeline the context fills up fast:

```python
import tiktoken

def audit_context_usage(
    system_prompt: str,
    chat_history: list[dict],
    retrieved_chunks: list[str],
    user_query: str,
    model: str = "gpt-4o",
) -> dict:
    """Break down token usage across all context components."""
    enc = tiktoken.get_encoding("o200k_base")

    def count(text: str) -> int:
        return len(enc.encode(text))

    history_text = " ".join(m["content"] for m in chat_history)
    chunks_text = "\n\n".join(retrieved_chunks)

    breakdown = {
        "system_prompt": count(system_prompt),
        "chat_history": count(history_text),
        "retrieved_chunks": count(chunks_text),
        "user_query": count(user_query),
    }
    breakdown["total_input"] = sum(breakdown.values())
    breakdown["remaining"] = 128_000 - breakdown["total_input"]  # GPT-4o
    return breakdown

# Example
result = audit_context_usage(
    system_prompt="You are a helpful assistant that answers questions from documents.",
    chat_history=[
        {"role": "user", "content": "What's the refund policy?"},
        {"role": "assistant", "content": "According to the document..."},
    ],
    retrieved_chunks=["Section 3.2: Refunds are processed within 14 days..."] * 5,
    user_query="Can I get a refund after 30 days?",
)

for key, val in result.items():
    print(f"{key:20}: {val:6,} tokens")
```

---

## The "lost in the middle" problem

Longer context sounds like a free lunch — just stuff everything in. But research shows model performance degrades on information placed in the **middle** of a long context.

```
Performance retrieving a fact from context:
Position: Beginning ████████████ 95%
Position: Middle    ██████       65%
Position: End       ████████████ 92%
```

This is the "lost in the middle" effect (Liu et al., 2023). It persists even in 1M-token models.

**Practical implication for RAG:**

```python
def reorder_chunks_for_attention(chunks: list[str], query: str) -> list[str]:
    """
    Put the most relevant chunks at the beginning and end.
    Bury less relevant chunks in the middle.
    This is sometimes called 'lost in the middle' mitigation.
    """
    # chunks already ranked by relevance score (highest first)
    if len(chunks) <= 2:
        return chunks

    # Most relevant → first, second most → last, rest in middle
    reordered = [chunks[0]] + chunks[2:] + [chunks[1]]
    return reordered

# In production: combine this with a reranker
# See: Week-02/Day-01-Part-2-Advanced-RAG
```

> [!warning] Big context ≠ good retrieval
> Sending 200 chunks to a 1M-token model is not better than sending the 5 most relevant chunks. The model's ability to precisely locate and use information degrades with context length. Use retrieval to narrow down, not to avoid retrieval.

---

## Context window vs. KV cache

The **KV cache** is a performance optimization, not a conceptual extension of the context window.

During inference, computing attention requires the Key and Value matrices of every previous token. Rather than recomputing these on every new token, the model caches them in GPU memory.

```
Without KV cache:
  Token 1000 generation: recompute K,V for tokens 1–999 → slow

With KV cache:
  Token 1000 generation: read cached K,V for tokens 1–999 → fast

Memory cost: d_model × seq_len × num_layers × 2 (K + V) × bytes_per_param
For Llama 3 70B, 128K context: ~32 GB just for the KV cache
```

> [!info] Prompt caching (Anthropic) and context caching (OpenAI)
> Both providers now cache prefix tokens across API calls. If your system prompt is 1,000 tokens and you make 1,000 calls, you pay for those tokens once, not 1,000 times. Cache hits are ~90% cheaper. Design your system prompts to be stable prefixes.

```python
# Anthropic prompt caching
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a helpful assistant.",
        },
        {
            "type": "text",
            "text": "LARGE_STABLE_DOCUMENT_HERE",  # 10,000 tokens
            "cache_control": {"type": "ephemeral"},  # cache this prefix
        },
    ],
    messages=[{"role": "user", "content": "Summarize section 3."}],
)

print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
# Second call: cache_read_tokens = 10,000, cost reduced by ~90%
```

---

## Practical guidelines

| Scenario | Recommendation |
|---------|---------------|
| RAG with many chunks | Keep total retrieved context < 20% of context window. Leave room for reasoning. |
| Long document Q&A | Use Claude Sonnet 4.6 (1M context) for docs < 700 pages. Chunk + summarize for larger. |
| Multi-turn chat | Truncate history after N turns, or summarize old turns. Track tokens actively. |
| Code generation | Reserve 4,096+ output tokens. Code responses are long. |
| Reasoning models (o3, o4) | These use token budget for chain-of-thought. Set `max_tokens` accordingly. |

```python
def sliding_window_history(
    messages: list[dict],
    max_history_tokens: int = 4000,
    model: str = "gpt-4o",
) -> list[dict]:
    """Trim chat history to stay within a token budget."""
    enc = tiktoken.get_encoding("o200k_base")

    total = 0
    trimmed = []
    for msg in reversed(messages):
        msg_tokens = len(enc.encode(msg["content"])) + 4
        if total + msg_tokens > max_history_tokens:
            break
        trimmed.insert(0, msg)
        total += msg_tokens

    return trimmed

# Always keep the most recent messages, drop the oldest
messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 50}
            for i in range(20)]
trimmed = sliding_window_history(messages, max_history_tokens=2000)
print(f"Kept {len(trimmed)} of {len(messages)} messages")
```

---

> [!success] Key takeaway
> The context window is your most precious resource. Monitor it, budget it, and always know how many tokens your system prompt, retrieved chunks, and chat history consume before the user even types a word.

[[02-tokenization]] | [[04-how-llms-generate-text]]
