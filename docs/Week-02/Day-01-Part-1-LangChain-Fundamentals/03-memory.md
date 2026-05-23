# Memory

Memory is what makes an LLM application feel like a conversation rather than a series of isolated requests. Without it, every turn starts fresh — the model doesn't know what was said two messages ago. LangChain provides several memory backends with different tradeoffs.

## Learning objectives

- Implement in-memory conversation history with `ChatMessageHistory`
- Apply token-aware windowing for long conversations
- Build a summary buffer that compresses old history
- Persist conversation history to a database for multi-session applications

---

## The memory problem

LLMs are stateless. Every API call is independent. Memory is the application-layer solution:

```
Turn 1: User: "My name is Alice."     → LLM: "Hello Alice!"
Turn 2: User: "What's my name?"       → LLM (no memory): "I don't know your name."
                                       → LLM (with memory): "Your name is Alice."

With memory: you pass all prior messages on every new call.
Without memory: you only pass the current message.
```

The challenge: conversation history grows with every turn. Eventually it exceeds the context window or becomes too expensive.

---

## ChatMessageHistory — in-memory history

The simplest pattern: store all messages in a list, pass them all on each call.

```python
import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

def create_chat_session(system_prompt: str = "You are a helpful assistant."):
    history = InMemoryChatMessageHistory()
    history.add_message(SystemMessage(content=system_prompt))

    def chat(user_message: str) -> str:
        history.add_message(HumanMessage(content=user_message))
        response = llm.invoke(history.messages)
        history.add_message(response)
        return response.content

    return chat, history

chat, history = create_chat_session("You are a concise coding assistant.")
print(chat("My name is Alice and I work in Python."))
print(chat("What language did I mention I work in?"))
print(chat("Suggest a Python library for data validation."))

# Inspect the full history
print(f"\nMessage count: {len(history.messages)}")
for msg in history.messages:
    print(f"  [{msg.__class__.__name__}]: {str(msg.content)[:60]}...")
```

---

## Token-aware windowing

As conversations grow, pass only the most recent N tokens to stay within context limits.

```python
import os
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import tiktoken

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
enc = tiktoken.encoding_for_model("gpt-4o-mini")

def count_message_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        total += 4  # overhead per message
        total += len(enc.encode(str(msg.content)))
    return total

def window_messages(
    messages: list[BaseMessage],
    max_tokens: int = 2000,
    keep_system: bool = True
) -> list[BaseMessage]:
    """Return a token-windowed subset of messages, always keeping the system message."""
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)] if keep_system else []
    conversation_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # Start from the most recent and work backwards
    windowed = []
    token_count = count_message_tokens(system_msgs)

    for msg in reversed(conversation_msgs):
        msg_tokens = count_message_tokens([msg])
        if token_count + msg_tokens > max_tokens:
            break
        windowed.insert(0, msg)
        token_count += msg_tokens

    return system_msgs + windowed

class WindowedChat:
    def __init__(self, system_prompt: str, max_tokens: int = 2000):
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self.max_tokens = max_tokens

    def chat(self, user_message: str) -> str:
        self.messages.append(HumanMessage(content=user_message))
        windowed = window_messages(self.messages, self.max_tokens)
        response = llm.invoke(windowed)
        self.messages.append(response)

        total = count_message_tokens(self.messages)
        windowed_count = count_message_tokens(windowed)
        print(f"  [tokens: total={total}, sent={windowed_count}]")

        return response.content

chat = WindowedChat("You are a helpful assistant.", max_tokens=1000)
for turn in [
    "Tell me about Python's history.",
    "What about its creator?",
    "What year was it first released?",
    "What is Python mainly used for today?",
    "What's its most popular library for data science?",
]:
    print(f"User: {turn}")
    print(f"AI: {chat.chat(turn)[:100]}...")
    print()
```

---

## Summary buffer memory

Summarize old conversation history instead of truncating it. Preserves semantic content at the cost of precision.

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

class SummaryBufferChat:
    """
    Keeps recent messages in full + a running summary of older messages.
    When total tokens exceed threshold, summarizes the oldest messages.
    """

    def __init__(
        self,
        system_prompt: str,
        max_recent_tokens: int = 1500,
        summary_model: str = "gpt-4o-mini"
    ):
        self.system = SystemMessage(content=system_prompt)
        self.summary: str = ""
        self.recent_messages: list = []
        self.max_recent_tokens = max_recent_tokens
        self.summary_llm = ChatOpenAI(
            model=summary_model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def _summarize(self, messages_to_summarize: list) -> str:
        if not messages_to_summarize:
            return self.summary

        conv_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in messages_to_summarize
        )

        prompt = f"""Current summary: {self.summary if self.summary else '(none)'}

New conversation to add to summary:
{conv_text}

Write a concise updated summary that captures the key facts and context from both."""

        response = self.summary_llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _build_context(self) -> list:
        messages = [self.system]
        if self.summary:
            messages.append(SystemMessage(content=f"Conversation summary so far: {self.summary}"))
        messages.extend(self.recent_messages)
        return messages

    def chat(self, user_message: str) -> str:
        self.recent_messages.append(HumanMessage(content=user_message))
        context = self._build_context()
        response = llm.invoke(context)
        self.recent_messages.append(response)

        # Check if we need to summarize old messages
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        recent_tokens = sum(len(enc.encode(str(m.content))) for m in self.recent_messages)

        if recent_tokens > self.max_recent_tokens:
            # Summarize the oldest half of messages
            cutoff = len(self.recent_messages) // 2
            to_summarize = self.recent_messages[:cutoff]
            self.summary = self._summarize(to_summarize)
            self.recent_messages = self.recent_messages[cutoff:]
            print(f"  [Summarized {cutoff} messages. Summary: {self.summary[:80]}...]")

        return response.content
```

---

## Persistent conversation history

For multi-session applications, persist message history to a database.

```python
import json
import os
from datetime import datetime
from pathlib import Path

class FileBackedChatHistory:
    """Simple file-backed conversation history for development use."""

    def __init__(self, session_id: str, storage_dir: str = "./chat_sessions"):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.file_path = self.storage_dir / f"{session_id}.json"
        self.messages = self._load()

    def _load(self) -> list[dict]:
        if self.file_path.exists():
            with open(self.file_path) as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        with open(self.file_path, "w") as f:
            json.dump(self.messages, f, indent=2, default=str)

    def add(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save()

    def to_langchain_messages(self):
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        role_map = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}
        return [role_map[m["role"]](content=m["content"])
                for m in self.messages if m["role"] in role_map]

# Usage
history = FileBackedChatHistory(session_id="user_alice_session_1")
history.add("system", "You are a helpful Python tutor.")
history.add("human", "What is a decorator?")
# history.add("ai", response_content)

# Next session — history persists across application restarts
same_history = FileBackedChatHistory(session_id="user_alice_session_1")
print(f"Loaded {len(same_history.messages)} messages from previous session")
```

> [!success] Production memory pattern
> For production: use Redis (`langchain_community.chat_message_histories.RedisChatMessageHistory`) or PostgreSQL (`langchain_community.chat_message_histories.PostgresChatMessageHistory`). Both support concurrent users and session expiration. File-backed is fine for development but not multi-process production.

---

[[02-chains-and-prompts]] | [[04-lcel]]
