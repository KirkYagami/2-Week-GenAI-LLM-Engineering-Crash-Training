# LangChain Overview

LangChain was created to solve a specific frustration: every LLM application was reinventing the same patterns — chat history, prompt templates, tool calling, retrieval. LangChain provides these abstractions once, in a way that works across model providers and deployment environments.

## Learning objectives

- Understand the core LangChain abstractions and when to use each
- Know the component hierarchy: models → prompts → chains → agents
- Set up LangChain with multiple model providers
- Know when to use LangChain vs raw API calls

---

## Core abstractions

```
LangChain component hierarchy:

┌─────────────────────────────────────────────┐
│  Agents                                      │
│  (LangGraph) — autonomous decision-making   │
│  [[../../../Week-02/Day-03-Part-2-LangGraph/01-langgraph-overview]]  │
└──────────────────┬──────────────────────────┘
                   │ uses
┌──────────────────▼──────────────────────────┐
│  Chains / LCEL                               │
│  Composable pipelines: prompt | llm | parser │
└──────────────────┬──────────────────────────┘
                   │ built from
┌──────────────────▼──────────────────────────┐
│  Core primitives                             │
│  Models, Prompts, Memory, Retrievers, Tools  │
└─────────────────────────────────────────────┘
```

### Models (ChatModels)

LangChain wraps all LLM providers under a common `ChatModel` interface. Swap providers by changing the class.

```python
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama

# All three have identical interfaces
gpt = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY")
)

claude = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0.0,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

local = Ollama(model="llama3.1:8b")  # Requires Ollama running locally

# Call any of them the same way
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LCEL in LangChain?")
]

# response = gpt.invoke(messages)
# response = claude.invoke(messages)
# response = local.invoke(messages)
# print(response.content)
```

### Prompt templates

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}")
])

# Fill in variables
prompt = chat_template.invoke({
    "role": "data scientist",
    "language": "English",
    "question": "What is cross-entropy loss?"
})
print(prompt.messages)
# [SystemMessage("You are a data scientist. Answer in English."),
#  HumanMessage("What is cross-entropy loss?")]
```

### Output parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# String parser — strips metadata from AIMessage
str_parser = StrOutputParser()

# Structured output parser with Pydantic
class ResumeScore(BaseModel):
    score: int = Field(description="Score 1-10")
    strengths: list[str] = Field(description="Key strengths")
    gaps: list[str] = Field(description="Missing skills")

json_parser = JsonOutputParser(pydantic_object=ResumeScore)
```

---

## When to use LangChain (and when not to)

| Use LangChain | Use raw API calls |
|---------------|------------------|
| Building multi-step pipelines | Simple one-off LLM calls |
| Switching between providers | Locked to one provider |
| Integrating retrievers, tools, memory | No retrieval or tool use |
| Rapid prototyping | Performance-critical production path |
| Reusing community integrations | Custom integrations only |

> [!warning] LangChain abstraction overhead
> LangChain adds a layer of abstraction that can obscure what's actually happening. If your pipeline is behaving unexpectedly, always check what prompt is actually being sent to the model. Use `langsmith` tracing or print the prompt before the LLM call to debug.

---

## LangChain ecosystem map

```python
# What lives where in the LangChain ecosystem:

# langchain-core       → base classes: Runnable, BaseMessage, ChatPromptTemplate
# langchain            → high-level chains, agents, memory (depends on langchain-core)
# langchain-openai     → ChatOpenAI, OpenAIEmbeddings
# langchain-anthropic  → ChatAnthropic
# langchain-community  → 100+ third-party integrations (Ollama, ChromaDB, etc.)
# langsmith            → tracing, evaluation, monitoring
# langgraph            → stateful agent graphs (Week 2 Day 3)

# Install only what you need
# pip install langchain langchain-openai langsmith
```

> [!tip] Import paths matter
> Always import from the most specific package. Import `ChatOpenAI` from `langchain_openai`, not `langchain.chat_models`. The latter is a re-export that may lag behind on updates and hides deprecation warnings.

---

[[00-agenda]] | [[02-chains-and-prompts]]
