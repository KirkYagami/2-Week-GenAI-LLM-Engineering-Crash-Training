# Agenda — Responsible AI and Safety

Deploying LLMs without safety measures is like deploying a web server without authentication. Your model will face adversarial inputs the moment it's public — from users probing its limits, from automated bots, and from genuinely harmful use cases. This session builds the practical skills to detect, prevent, and respond to these failure modes.

## Learning objectives

By the end of this session you will be able to:

- Identify and defend against jailbreaks and prompt injection attacks
- Implement input/output guardrails using moderation APIs and custom classifiers
- Apply content filtering at multiple pipeline stages
- Audit your system for demographic bias and fairness issues

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:35 | Jailbreaks and prompt injection | [[01-jailbreaks-and-prompt-injection]] |
| 0:35 – 1:10 | Guardrails — input validation and output checks | [[02-guardrails]] |
| 1:10 – 1:40 | Content filtering APIs and custom classifiers | [[03-content-filtering]] |
| 1:40 – 2:20 | Bias, fairness, and demographic auditing | [[04-bias-and-fairness]] |
| 2:20 – 3:00 | Practice exercises | [[05-practice-exercises]] |

## Setup

```bash
pip install openai anthropic presidio-analyzer presidio-anonymizer transformers
```

```python
import os
from openai import OpenAI
from anthropic import Anthropic

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

> [!info] Why responsible AI is not optional
> In the EU, the AI Act classifies LLMs deployed in high-risk domains (employment, credit, healthcare) under strict requirements for bias auditing, documentation, and human oversight. In the US, executive orders require federal agencies to assess AI risks. Beyond regulation: a single high-profile safety failure can end a product faster than a bad review.

[[../Day-04-Part-1-LLM-Evaluation/07-interview-questions|← Day 04 Part 1]] | [[01-jailbreaks-and-prompt-injection|Start →]]
