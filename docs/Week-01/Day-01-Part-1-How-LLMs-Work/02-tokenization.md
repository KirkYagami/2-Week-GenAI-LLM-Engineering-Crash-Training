# Tokenization

LLMs don't read words. They read **tokens** — chunks of text that sit somewhere between a character and a word. Understanding tokenization directly affects your prompt costs, failure modes, and retrieval pipeline design.

## Learning objectives

- Explain Byte Pair Encoding (BPE) and why it's the dominant algorithm
- Count tokens accurately for any OpenAI or Anthropic model
- Predict tokenization failures before they hit production
- Estimate API costs from token counts

---

## What is a token?

A token is the atomic unit of text that an LLM processes. Depending on the language and character, a token is roughly:

- **1 token ≈ 4 characters** in English
- **1 token ≈ 0.75 words** in English
- **1 token ≈ 1 character** in Chinese, Japanese, or Korean (non-Latin scripts tokenize less efficiently)

```python
# Concrete examples using tiktoken (OpenAI's tokenizer)
import tiktoken

enc = tiktoken.get_encoding("o200k_base")  # GPT-4o, o1, o3, o4-mini

examples = [
    "Hello",          # 1 token
    "Hello, world!",  # 4 tokens
    "tokenization",   # 1 token (common word)
    "tokenizationnnn", # 3 tokens (uncommon suffix)
    "1234567890",     # varies — digits often split
    "한국어",           # 4 tokens (Korean is less efficient)
]

for text in examples:
    tokens = enc.encode(text)
    print(f"{text!r:30} → {len(tokens):2d} tokens: {tokens}")
```

> [!warning] Non-English text costs more
> Code in Python: ~1.3 tokens/word. Code in Korean or Arabic: 2–4× more tokens for the same semantic content. Always count tokens for your actual language before estimating costs.

---

## Byte Pair Encoding (BPE)

BPE is the algorithm behind tiktoken (OpenAI) and most modern LLM tokenizers. The intuition:

1. Start with a vocabulary of individual characters (a, b, c, …, 0, 1, …)
2. Find the most frequent pair of adjacent tokens in your training corpus
3. Merge that pair into a new token and add it to the vocabulary
4. Repeat until you reach the target vocabulary size (e.g., 100,000 tokens)

The result: common words ("the", "and", "is") become single tokens. Rare words split into recognizable subwords ("transformer" → "transform" + "er"). Unknown words never fail — they always fall back to individual bytes.

```python
# Build a minimal BPE tokenizer from scratch
from collections import Counter

def get_stats(vocab):
    """Count frequency of every adjacent pair in the vocabulary."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge the best pair across all words."""
    import re
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    return {pattern.sub("".join(pair), word): freq for word, freq in vocab.items()}

# Toy corpus
vocab = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}

print("Initial vocab:", vocab)
for i in range(5):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"Merge {i+1}: {best} → {''.join(best)}")

# Output shows: es, est, lo, low, ne being merged
```

---

## Tiktoken: OpenAI's tokenizer

Tiktoken is OpenAI's production tokenizer, written in Rust for speed (3–6× faster than Python alternatives).

```python
import tiktoken

# Model-to-encoding mapping (2025)
MODEL_ENCODINGS = {
    "gpt-4o": "o200k_base",         # 200k vocab
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-4": "cl100k_base",         # 100k vocab (older)
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding_name = MODEL_ENCODINGS.get(model, "o200k_base")
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

# Count tokens in a chat message list (mirrors the actual API cost)
def count_chat_tokens(messages: list, model: str = "gpt-4o") -> int:
    enc = tiktoken.get_encoding(MODEL_ENCODINGS.get(model, "o200k_base"))
    total = 0
    for msg in messages:
        total += 4  # every message has overhead: role, separators
        total += len(enc.encode(msg["content"]))
    total += 2  # reply priming
    return total

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain transformers in two sentences."},
]

print(f"Total tokens: {count_chat_tokens(messages)}")
# Expected: ~25 tokens
```

> [!info] Vocabulary size matters
> GPT-4o uses `o200k_base` (200K vocab) vs GPT-4's `cl100k_base` (100K vocab). The larger vocabulary means more words tokenize as a single token — slightly cheaper and slightly faster for the same content.

---

## Anthropic's tokenizer

Claude uses its own tokenizer, not tiktoken. Use the official Anthropic client to count tokens.

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

def count_anthropic_tokens(messages: list, model: str = "claude-sonnet-4-6") -> int:
    """Count tokens before making an API call — avoids surprise bills."""
    response = client.messages.count_tokens(
        model=model,
        messages=messages,
    )
    return response.input_tokens

messages = [{"role": "user", "content": "Explain transformers in two sentences."}]
token_count = count_anthropic_tokens(messages)
print(f"Input tokens: {token_count}")

# For system prompt + user message:
response = client.messages.count_tokens(
    model="claude-sonnet-4-6",
    system="You are a helpful assistant.",
    messages=messages,
)
print(f"With system prompt: {response.input_tokens} tokens")
```

> [!tip] Always count before expensive calls
> Before sending a 100-page document to the API, count its tokens. Claude Sonnet 4.6 costs $3/million input tokens. A 500-page PDF ≈ 150,000 tokens ≈ $0.45 per call. Count first, chunk if needed.

---

## Tokenization failure modes

These bite engineers in production. Know them before they hit you.

### 1. Number and arithmetic problems

```python
enc = tiktoken.get_encoding("o200k_base")

# Numbers often split in unexpected ways
for n in ["100", "1000", "10000", "1234567"]:
    tokens = enc.encode(n)
    print(f"{n!r} → {len(tokens)} tokens: {[enc.decode([t]) for t in tokens]}")
# '100' → 1 token
# '10000' → 2 tokens: ['100', '00']
# '1234567' → 3 tokens: ['123', '456', '7']
```

This is why LLMs struggle with arithmetic — the model never sees the number "1234567" as a single unit. Always use tool calls or code execution for math.

### 2. The tokenization boundary problem in RAG

```python
# BAD: splitting on character count ignores token boundaries
text = "The transformer architecture revolutionized NLP."
bad_chunk = text[:20]  # "The transformer archi" — cuts mid-token

# GOOD: split on token boundaries
enc = tiktoken.get_encoding("o200k_base")
tokens = enc.encode(text)
good_chunk = enc.decode(tokens[:10])  # exactly 10 tokens
print(good_chunk)
```

### 3. Special tokens

```python
# These tokens have special meaning — don't put them in user content
SPECIAL_TOKENS = {
    "<|endoftext|>": "Signals end of document in GPT models",
    "<|fim_prefix|>": "Fill-in-the-Middle prefix",
    "<|fim_suffix|>": "Fill-in-the-Middle suffix",
}

# tiktoken raises an error if you try to encode special tokens by accident
try:
    enc.encode("<|endoftext|>")
except Exception as e:
    print(f"Error: {e}")
    # Use: enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
```

> [!warning] Never trust character counts for cost estimation
> Character count / 4 is a rough approximation. For budget-critical applications, always use the actual tokenizer. Off-by-2× errors are common with code, markdown, and non-English text.

---

## Cost estimation cheat sheet

| Model | Input cost (per 1M tokens) | Output cost (per 1M tokens) |
|-------|--------------------------|---------------------------|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| o3 | $10.00 | $40.00 |
| o4-mini | $1.10 | $4.40 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $0.80 | $4.00 |

*Prices as of May 2026. Always verify at platform.openai.com/pricing and anthropic.com/pricing.*

```python
def estimate_cost(prompt: str, response_tokens: int, model: str = "gpt-4o") -> float:
    """Rough cost estimate for a single API call."""
    PRICING = {
        "gpt-4o":       (2.50, 10.00),
        "gpt-4o-mini":  (0.15, 0.60),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5":  (0.80, 4.00),
    }
    input_price, output_price = PRICING.get(model, (3.00, 15.00))
    input_tokens = count_tokens(prompt, model)
    cost = (input_tokens / 1_000_000) * input_price
    cost += (response_tokens / 1_000_000) * output_price
    return cost

prompt = "Summarize the following document: " + "word " * 1000
print(f"Estimated cost: ${estimate_cost(prompt, response_tokens=200):.4f}")
# Expected: ~$0.008 for gpt-4o
```

---

## Practice

1. Install tiktoken and count the tokens in your last 10 WhatsApp messages. Is the 4-chars-per-token approximation accurate?
2. Tokenize the same sentence in English, German, and Chinese. How does token count differ?
3. Find a number that tokenizes into 3+ tokens with `o200k_base`. Explain why.

---

> [!success] Key takeaway
> Tokens are the unit of cost, context, and computation. Every time you design a prompt, retrieve a chunk, or plan an API budget, you're reasoning about tokens. Master the tokenizer — it pays off immediately.

[[01-transformers-and-attention]] | [[03-context-windows]]
