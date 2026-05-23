# Agenda — OpenAI and Anthropic APIs

Two frontier API providers with different design philosophies but overlapping capability. Today you learn both well enough to build production systems with either.

## Learning objectives

By the end of this session you will be able to:

- Make authenticated API calls to OpenAI and Anthropic from Python
- Pass images and documents through the vision/multimodal APIs
- Implement tool use (function calling) in both APIs
- Estimate cost and respect rate limits before shipping
- Choose between the two providers based on task requirements

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:30 | OpenAI Chat Completions — parameters, streaming, async | [[01-openai-chat-completions]] |
| 0:30 – 1:00 | Vision and Multimodal — images, PDFs, structured extraction | [[02-vision-and-multimodal]] |
| 1:00 – 1:30 | Tool Use with OpenAI — parallel tool calls, strict schemas | [[03-tool-use-openai]] |
| 1:30 – 2:00 | Anthropic Messages API — differences, tool_use, streaming | [[04-anthropic-messages-api]] |
| 2:00 – 2:30 | Cost and Rate Limits — tokens, pricing, retry logic | [[05-cost-and-rate-limits]] |
| 2:30 – 3:00 | Practice Exercises + Q&A | [[06-practice-exercises]] |

## Setup

```python
import os
from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Quick smoke test
r = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say 'API working' and nothing else."}],
    max_tokens=10
)
print(r.choices[0].message.content)
```

> [!tip] Get API keys
> OpenAI: platform.openai.com → API Keys. Anthropic: console.anthropic.com → API Keys.
> Store both in a `.env` file and load with `python-dotenv`. Never commit `.env` to git.

[[../Day-01-Part-2-Prompt-Engineering/07-interview-questions|← Day 01 Part 2]] | [[01-openai-chat-completions|Start →]]
