# Agenda — Prompt Engineering

**Session length:** ~3 hours | **Difficulty:** Beginner → Intermediate | **Coding time:** ~1.5 hours

## Why this session matters

Prompt engineering is the highest-leverage skill in LLM application development. A well-crafted prompt turns a mediocre model into an excellent one — and a poorly written prompt turns an excellent model into a mediocre tool. Before reaching for fine-tuning or RAG, exhaust what good prompting can do.

## Learning objectives

By the end of Part 2 you will be able to:

- Write zero-shot and few-shot prompts that produce consistent, structured output
- Apply chain-of-thought reasoning to improve accuracy on multi-step tasks
- Design system prompts that shape model persona, scope, and output format
- Extract structured data using both prompt-based and API-native methods
- Recognize and fix the 10 most common prompt failure modes

## Session outline

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:35 | Zero-shot and few-shot prompting | [[01-zero-shot-and-few-shot]] |
| 0:35 – 1:10 | Chain-of-thought prompting | [[02-chain-of-thought]] |
| 1:10 – 1:40 | Structured output — JSON and schemas | [[03-structured-output]] |
| 1:40 – 2:10 | System prompts — design and gotchas | [[04-system-prompts]] |
| 2:10 – 2:30 | Advanced prompt patterns | [[05-prompt-patterns]] |
| 2:30 – 2:45 | Practice exercises | [[06-practice-exercises]] |
| 2:45 – 3:00 | Interview questions review | [[07-interview-questions]] |

## Prerequisites

- Completed Day 1 Part 1 (How LLMs Work)
- OpenAI and/or Anthropic API key

## Setup

```bash
pip install openai anthropic pydantic
```

```python
import os
import openai
import anthropic

openai_client = openai.OpenAI()        # reads OPENAI_API_KEY
anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
```

> [!tip] The prompt engineering mindset
> Think of the model as an extremely capable intern on their first day. They're smart, they want to help, they'll do exactly what you say — including the parts you didn't intend. Be explicit, be concrete, and always specify what you *don't* want as well as what you do.

---

[[Week-01/Day-01-Part-1-How-LLMs-Work/06-interview-questions]] | Next: [[01-zero-shot-and-few-shot]]
