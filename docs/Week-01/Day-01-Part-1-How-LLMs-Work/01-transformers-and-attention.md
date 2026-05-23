# Transformers and Attention

Before transformers, language models processed text one word at a time — slow, forgetful, and unable to scale. The 2017 paper "Attention Is All You Need" replaced recurrence with a single mechanism: **self-attention**. Every modern LLM — GPT-4o, Claude, Llama, Gemini — is a transformer.

## Learning objectives

- Understand why transformers replaced RNNs for language modeling
- Explain self-attention, multi-head attention, and positional encoding
- Map the data flow through a transformer block
- Name the key architectural variants used in 2025 production models

---

![Transformer architecture — decoder-only stack](../../../drawings/transformer-architecture.excalidraw)

---

## Why transformers replaced RNNs

Recurrent Neural Networks (RNNs) processed tokens sequentially: word 1 → word 2 → word 3. Two problems:

1. **Can't parallelize training** — each step depends on the previous one
2. **Vanishing gradient** — information from early tokens fades over long sequences

Transformers process the *entire sequence simultaneously*. Token 1 and token 500 are computed at the same time. This is why you can train a 70B-parameter model on thousands of GPUs — the work is embarrassingly parallel.

---

## The transformer block

A transformer model is a stack of identical layers. Each layer has two sub-components:

```
Input Tokens
    ↓
[Embedding + Positional Encoding]
    ↓
┌─────────────────────────────┐
│  Multi-Head Self-Attention  │  ← "Who should I pay attention to?"
│  + Add & LayerNorm          │
├─────────────────────────────┤
│  Feed-Forward Network       │  ← "What do I think about this?"
│  + Add & LayerNorm          │
└─────────────────────────────┘
    ↓
[Repeat N times — GPT-4 has ~96 layers]
    ↓
Linear + Softmax → Next Token Probabilities
```

---

## Self-attention: the core idea

Self-attention lets every token ask: *"Which other tokens in this sequence are most relevant to understanding me right now?"*

It does this with three learned matrices applied to each token embedding:

| Matrix | Role | Analogy |
|--------|------|---------|
| **Q** (Query) | What am I looking for? | A search query |
| **K** (Key) | What do I offer? | A document title |
| **V** (Value) | What do I actually contain? | The document body |

The attention score between token `i` and token `j`:

```
score(i, j) = softmax( Q_i · K_j / √d_k )
```

- `d_k` is the key dimension (scaling prevents exploding gradients in large models)
- Softmax turns raw scores into probabilities that sum to 1
- The output for token `i` is a weighted sum of all Value vectors

**Example:** In the sentence *"The bank by the river flooded"*, the word "bank" attends strongly to "river" and "flooded" to resolve the ambiguity — financial institution or riverbank.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """Core attention operation — same math as every transformer."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Toy example: 4 tokens, 8-dim embeddings
seq_len, d_model = 4, 8
Q = torch.randn(seq_len, d_model)
K = torch.randn(seq_len, d_model)
V = torch.randn(seq_len, d_model)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")        # (4, 8)
print(f"Attention weights:\n{attn_weights}")   # 4×4 matrix
```

---

## Multi-head attention

Running attention once gives one "perspective." Running it `h` times in parallel — each with different Q/K/V projections — lets the model capture multiple types of relationships simultaneously:

- Head 1: syntactic dependencies ("subject → verb")
- Head 2: coreference ("she" → "Alice")
- Head 3: semantic similarity
- Head N: whatever the model learned was useful

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        # Project and split into heads
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Attention per head
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)

# GPT-3 scale: 96 layers, 96 heads, 12288-dim embeddings
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq=10, d_model=512
print(mha(x).shape)  # (2, 10, 512)
```

---

## Positional encoding

Self-attention treats the sequence as a *bag of tokens* — it has no built-in sense of word order. Positional encoding adds order information to each token embedding.

**Two strategies in production models:**

| Strategy | Used by | How it works |
|----------|---------|--------------|
| **Sinusoidal (absolute)** | Original transformer, BERT | Fixed sine/cosine patterns added to embeddings |
| **RoPE (Rotary Position Embedding)** | Llama 3, Mistral, GPT-NeoX | Rotates Q and K vectors — enables longer context extrapolation |

RoPE is now the dominant approach because it generalizes better to sequences longer than those seen during training — crucial for the 128K–1M context windows of modern models.

---

## The feed-forward network

After attention, each token passes through a two-layer MLP independently:

```
FFN(x) = GELU( x · W₁ + b₁ ) · W₂ + b₂
```

The hidden dimension is typically 4× the model dimension. In GPT-3 (12,288-dim), the FFN hidden layer has ~49,152 neurons. This is where most of the model's parameters live — and where factual knowledge is thought to be stored.

> [!info] The "knowledge" question
> Research suggests attention layers learn *relationships* (syntax, coreference) while FFN layers store *facts* (Paris is the capital of France). This is why fine-tuning often focuses on FFN weights.

---

## 2025 architectural variants

Production LLMs have evolved beyond the original transformer:

| Innovation | What it does | Models using it |
|-----------|--------------|-----------------|
| **GQA (Grouped-Query Attention)** | Multiple query heads share one K/V head — reduces memory during inference | Llama 3, Mistral, GPT-4o |
| **MLA (Multi-Head Latent Attention)** | Compresses K/V into a low-rank latent space — shrinks KV cache by 5–13× | DeepSeek-V3, DeepSeek-R1 |
| **FlashAttention-3** | Reorders GPU memory accesses for 2–3× speedup — no change to output | Used in most production inference stacks |
| **Mixture of Experts (MoE)** | Only a subset of FFN layers ("experts") activate per token — sparse computation | GPT-4, Mixtral 8×7B, DeepSeek |
| **SwiGLU activation** | Replaces GELU in FFN — better loss curves | Llama, PaLM, Gemma |

> [!tip] Why GQA matters for you
> GQA directly affects inference cost. A model with GQA can serve longer contexts with less GPU memory — which is why Llama 3 70B can run on a single A100 where earlier 70B models could not.

---

## Encoder-only vs decoder-only vs encoder-decoder

| Architecture | Training objective | Best for | Examples |
|---|---|---|---|
| **Encoder-only** | Masked language modeling (predict masked tokens) | Classification, embeddings, search | BERT, RoBERTa |
| **Decoder-only** | Causal language modeling (predict next token) | Text generation, chat, reasoning | GPT-4o, Claude, Llama 3 |
| **Encoder-decoder** | Seq2seq (encode input, decode output) | Translation, summarization | T5, BART, Flan-T5 |

All the models you'll use in this course — GPT-4o, Claude, Llama 3 — are **decoder-only**. The encoder is not needed when the model sees the full input before generating.

> [!warning] Common misconception
> "ChatGPT uses BERT" — No. BERT is encoder-only and can't generate text. GPT-4o is decoder-only. They're fundamentally different architectures despite both being transformers.

---

## Key numbers to know

| Model | Layers | Heads | d_model | Parameters |
|-------|--------|-------|---------|------------|
| GPT-2 | 12 | 12 | 768 | 117M |
| GPT-3 | 96 | 96 | 12,288 | 175B |
| Llama 3 8B | 32 | 32 | 4,096 | 8B |
| Llama 3 70B | 80 | 64 | 8,192 | 70B |

Bigger is not always better — Llama 3 8B outperforms GPT-3 on most benchmarks despite being 20× smaller, thanks to better training data and RLHF.

---

## Common mistakes

> [!warning] Attention is O(n²) in sequence length
> The attention matrix is `seq_len × seq_len`. Doubling the context window quadruples the memory. This is why FlashAttention and GQA exist — and why you should profile your RAG chunk sizes carefully.

> [!warning] More heads ≠ better
> Increasing the number of attention heads beyond the model's capacity wastes compute. The optimal head count depends on d_model and data. Don't tune this unless you're training from scratch.

---

## What's next

Understanding attention is necessary but not sufficient — you also need to understand how the model reads your input. That starts with tokenization.

[[02-tokenization]] | [[00-agenda]]
