# Agenda — How LLMs Work

**Session length:** ~3 hours | **Difficulty:** Beginner → Intermediate | **Coding time:** ~1 hour

## Why this session matters

Every practical decision you make in this course — which model to pick, how to set temperature, why your 500-page PDF retrieval fails — traces back to one question: *what is actually happening inside the model?* This session answers that question with enough depth to make you dangerous, without requiring a PhD in mathematics.

## Learning objectives

By the end of Part 1 you will be able to:

- Explain how the transformer architecture processes a sentence in parallel
- Describe what attention scores represent and why they matter
- Trace the journey from raw text → tokens → embeddings → next-token prediction
- Reason about context window limits when designing RAG pipelines
- Tune `temperature`, `top_p`, and `top_k` with confidence for any task

## Session outline

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:40 | Transformers and the attention mechanism | [[01-transformers-and-attention]] |
| 0:40 – 1:10 | Tokenization — how text becomes numbers | [[02-tokenization]] |
| 1:10 – 1:30 | Context windows — what the model can "see" | [[03-context-windows]] |
| 1:30 – 2:00 | How LLMs generate text — sampling and decoding | [[04-how-llms-generate-text]] |
| 2:00 – 2:30 | Hands-on practice exercises | [[05-practice-exercises]] |
| 2:30 – 3:00 | Interview questions review | [[06-interview-questions]] |

## Prerequisites

None. This is Day 1. Bring Python 3.10+ and a terminal.

## Setup

```bash
pip install openai anthropic tiktoken
```

```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
```

> [!tip] Before the session
> Skim the abstract of "Attention Is All You Need" (Vaswani et al., 2017). Don't worry about the math — just absorb the vocabulary. Every concept we cover today traces back to that eight-page paper.

---

[[Week-01/README]] | Next: [[01-transformers-and-attention]]
