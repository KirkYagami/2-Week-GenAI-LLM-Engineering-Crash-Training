# Agent Memory Strategies

An agent's context window is its working memory. After 10–20 steps, tool outputs and prior messages fill it up, causing the model to lose track of earlier context or hit token limits. Memory strategies control what an agent remembers and how.

## Learning objectives

- Implement three memory strategies: in-context, summarized, and vector-based
- Apply windowing to prevent context overflow
- Summarize old context to preserve key facts
- Store and retrieve relevant memories from a vector store
- Choose the right memory strategy for the task

---

## Strategy 1: In-context (full history)

The simplest approach: keep every message. Works for short tasks; fails as conversations grow.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class FullContextAgent:
    def __init__(self, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=self.messages, max_tokens=500,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        self.total_tokens += response.usage.total_tokens
        return reply

    @property
    def context_length(self) -> int:
        return sum(len(m["content"].split()) * 1.3 for m in self.messages)

# Works fine for short sessions
agent = FullContextAgent("You are a helpful research assistant.")
print(agent.chat("What is machine learning?"))
print(f"Approx tokens in context: {agent.context_length:.0f}")
```

**Limit:** gpt-4o-mini has a 128K context window. At ~500 tokens per exchange, that's ~256 turns before truncation. Fine for most tasks; problematic for long-running sessions.

---

## Strategy 2: Sliding window

Keep only the last N messages. Fast, but loses older context completely.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class WindowedAgent:
    def __init__(self, system_prompt: str, window_size: int = 10):
        self.system_prompt = system_prompt
        self.history = []  # Only user/assistant messages
        self.window_size = window_size  # Number of turns to keep

    def _build_messages(self) -> list[dict]:
        # Always include system prompt + recent window
        recent = self.history[-self.window_size * 2:]  # *2 for user+assistant pairs
        return [{"role": "system", "content": self.system_prompt}] + recent

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        messages = self._build_messages()

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=500,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

# After many turns, older messages are dropped
agent = WindowedAgent("You are a helpful research assistant.", window_size=5)
for q in ["What is Python?", "Tell me about data types.", "What about lists?", "And dicts?"]:
    agent.chat(q)
print(f"History length: {len(agent.history)}, Context messages: {len(agent._build_messages())}")
```

---

## Strategy 3: Summary buffer

When history exceeds a threshold, summarize older messages and keep the summary + recent messages. Preserves key facts while controlling token cost.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SummaryBufferAgent:
    def __init__(self, system_prompt: str, max_recent_tokens: int = 2000):
        self.system_prompt = system_prompt
        self.summary = ""          # Compressed history
        self.recent = []           # Recent messages (full)
        self.max_recent_tokens = max_recent_tokens

    def _estimate_tokens(self, messages: list[dict]) -> int:
        return int(sum(len(m["content"].split()) * 1.3 for m in messages))

    def _compress(self) -> None:
        """Summarize oldest half of recent messages into the running summary."""
        if len(self.recent) < 4:
            return

        to_compress = self.recent[:len(self.recent) // 2]
        conversation_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in to_compress
        )
        prompt = f"""Summarize this conversation, keeping key facts, decisions, and context.
Existing summary: {self.summary or 'None'}
New conversation:
{conversation_text}

Updated summary (be concise, max 200 words):"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=300,
        )
        self.summary = resp.choices[0].message.content
        self.recent = self.recent[len(self.recent) // 2:]
        print(f"  [Compressed: summary={len(self.summary.split())} words, recent={len(self.recent)} msgs]")

    def _build_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation:\n{self.summary}"
            })
        messages.extend(self.recent)
        return messages

    def chat(self, user_message: str) -> str:
        self.recent.append({"role": "user", "content": user_message})

        # Compress if recent messages are too long
        if self._estimate_tokens(self.recent) > self.max_recent_tokens:
            self._compress()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self._build_messages(),
            max_tokens=500,
        )
        reply = response.choices[0].message.content
        self.recent.append({"role": "assistant", "content": reply})
        return reply

# Test over many turns
agent = SummaryBufferAgent("You are a research assistant.", max_recent_tokens=500)
topics = ["What is Python?", "What are its main use cases?", "What is numpy?",
          "What is pandas?", "What is scikit-learn?", "How do these tools work together?"]

for t in topics:
    reply = agent.chat(t)
    print(f"Q: {t}")
    print(f"A: {reply[:100]}...\n")
```

---

## Strategy 4: Vector-based long-term memory

Store important facts in a vector database; retrieve relevant memories at each step.

```python
import os
import numpy as np
from openai import OpenAI
from dataclasses import dataclass, field

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class VectorMemoryAgent:
    system_prompt: str
    recent_window: int = 6
    memory_store: list[dict] = field(default_factory=list)  # [{text, embedding}]
    recent: list[dict] = field(default_factory=list)

    def _embed(self, text: str) -> np.ndarray:
        resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
        emb = np.array(resp.data[0].embedding)
        return emb / np.linalg.norm(emb)

    def store_memory(self, text: str) -> None:
        """Store a fact or observation in long-term memory."""
        emb = self._embed(text)
        self.memory_store.append({"text": text, "embedding": emb})

    def retrieve_memories(self, query: str, k: int = 3) -> list[str]:
        """Retrieve top-k most relevant memories."""
        if not self.memory_store:
            return []
        q_emb = self._embed(query)
        scores = [(m["text"], float(q_emb @ m["embedding"])) for m in self.memory_store]
        scores.sort(key=lambda x: -x[1])
        return [text for text, _ in scores[:k]]

    def chat(self, user_message: str) -> str:
        # Retrieve relevant memories
        memories = self.retrieve_memories(user_message)

        messages = [{"role": "system", "content": self.system_prompt}]
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            messages.append({"role": "system", "content": f"Relevant memories:\n{memory_text}"})

        messages.extend(self.recent[-self.recent_window * 2:])
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=500,
        )
        reply = response.choices[0].message.content
        self.recent.append({"role": "user", "content": user_message})
        self.recent.append({"role": "assistant", "content": reply})
        return reply

# Test
agent = VectorMemoryAgent("You are a personalized research assistant.")
agent.store_memory("User is a Python developer with 5 years of experience.")
agent.store_memory("User prefers concise answers with code examples.")
agent.store_memory("User is working on a recommendation system project.")

print(agent.chat("Suggest a good approach for my project."))
```

---

## Choosing a memory strategy

| Strategy | Token cost | Preserves history | Best for |
|----------|-----------|-----------------|----------|
| Full context | High | Yes (until limit) | Short tasks, few turns |
| Sliding window | Low | Last N turns only | Chat assistants, REPL agents |
| Summary buffer | Medium | Compressed | Long research sessions |
| Vector memory | Low + embedding | Relevant facts | Personalized assistants, long-running agents |

> [!success] Combine summary buffer + vector memory for production agents
> Summary buffer handles the recent conversation flow. Vector memory stores durable facts (user preferences, prior decisions, domain knowledge). Together, they give you full control over what the agent remembers without hitting context limits.

---

[[04-tool-calling-agents]] | [[06-practice-exercises]]
