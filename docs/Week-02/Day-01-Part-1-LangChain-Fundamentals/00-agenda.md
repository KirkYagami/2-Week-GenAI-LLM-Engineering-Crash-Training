# Agenda — LangChain Fundamentals

LangChain is the most widely used framework for building LLM applications. It solves two problems: composability (chaining LLM calls, tools, and data sources) and portability (switching between OpenAI, Anthropic, Ollama, and others without rewriting your code). In Week 2, it's your primary building block.

## Learning objectives

By the end of this session you will be able to:

- Build prompt templates and chains using LangChain
- Implement conversation memory with multiple backends
- Write composable pipelines using LCEL (LangChain Expression Language)
- Swap model providers without changing business logic

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:20 | LangChain overview — why it exists, key abstractions | [[01-langchain-overview]] |
| 0:20 – 1:00 | Chains and prompts — templates, few-shot, output parsers | [[02-chains-and-prompts]] |
| 1:00 – 1:35 | Memory — conversation history and context management | [[03-memory]] |
| 1:35 – 2:15 | LCEL — composable pipelines with the pipe operator | [[04-lcel]] |
| 2:15 – 3:00 | Practice exercises | [[05-practice-exercises]] |

## Setup

```bash
pip install langchain langchain-openai langchain-anthropic langchain-community langsmith
```

```python
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Initialize models — same interface, different provider
gpt = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
claude = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY"))
```

> [!info] LangChain version
> This session uses LangChain 0.3.x with LCEL as the primary composition pattern. The older "chain" classes (`LLMChain`, `SequentialChain`) still work but are superseded by LCEL. New code should use LCEL pipes.

[[../../../Week-01/Day-05-Part-2-Local-LLMs/07-interview-questions|← Week 1 Day 5]] | [[01-langchain-overview|Start →]]
