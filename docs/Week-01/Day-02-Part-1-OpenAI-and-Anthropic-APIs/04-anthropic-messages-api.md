# Anthropic Messages API

The Anthropic Messages API powers Claude. It shares the same core concepts as OpenAI (messages, tools, streaming) but has important design differences that affect how you write production code.

## Learning objectives

- Understand the structural differences between OpenAI and Anthropic APIs
- Use Claude's tool_use blocks correctly
- Implement streaming with event-based parsing
- Apply prompt caching to reduce cost on long system prompts

---

## API structure differences at a glance

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System prompt | `messages[0].role = "system"` | Separate `system` parameter |
| Response object | `response.choices[0].message.content` | `response.content` (list of blocks) |
| Tool calls | `message.tool_calls` (list) | `content` blocks with `type: "tool_use"` |
| Tool results | `role: "tool"` messages | `role: "user"` with `type: "tool_result"` block |
| Streaming | `stream=True`, delta chunks | Event-based stream with typed events |
| Context window | 128K (gpt-4o) | 200K (claude-sonnet-4-6) |
| Token counting | `response.usage` | `response.usage` + separate `client.messages.count_tokens()` |

---

## Basic messages API call

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    system="You are a concise technical assistant. Answer in 2-3 sentences.",
    messages=[
        {"role": "user", "content": "What is the difference between RAG and fine-tuning?"}
    ]
)

print(response.content[0].text)
print(f"\nInput tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
```

The `content` field is a list of `ContentBlock` objects. For text-only responses, `content[0].text` is always safe. For tool use, you may have multiple blocks of different types.

---

## Multi-turn conversations

```python
def chat(client: Anthropic, system: str, messages: list[dict], user_input: str) -> tuple[str, list[dict]]:
    messages = messages + [{"role": "user", "content": user_input}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=system,
        messages=messages
    )

    assistant_reply = response.content[0].text
    messages = messages + [{"role": "assistant", "content": assistant_reply}]
    return assistant_reply, messages

# Usage
system = "You are a helpful Python tutor."
history = []

reply, history = chat(client, system, history, "What is a decorator?")
print(f"Claude: {reply}\n")

reply, history = chat(client, system, history, "Show me a simple example.")
print(f"Claude: {reply}\n")
```

---

## Tool use with Claude

Claude's tool use follows the same logical flow as OpenAI but uses different data structures.

```python
import json

tools = [
    {
        "name": "get_customer",
        "description": "Retrieve customer information by customer ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "The customer's unique identifier"
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "update_subscription",
        "description": "Update a customer's subscription plan.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "plan": {"type": "string", "enum": ["free", "pro", "enterprise"]}
            },
            "required": ["customer_id", "plan"]
        }
    }
]

# Fake implementations
def get_customer(customer_id: str) -> dict:
    return {"id": customer_id, "name": "Alice Johnson", "plan": "free", "email": "alice@example.com"}

def update_subscription(customer_id: str, plan: str) -> dict:
    return {"success": True, "customer_id": customer_id, "new_plan": plan}

TOOL_REGISTRY = {"get_customer": get_customer, "update_subscription": update_subscription}

def run_claude_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            # Extract text from final response
            return next(b.text for b in response.content if b.type == "text")

        if response.stop_reason != "tool_use":
            break

        # Append Claude's response (contains tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call and collect results
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            fn = TOOL_REGISTRY[block.name]
            result = fn(**block.input)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)
            })

        # Send results back as a user message
        messages.append({"role": "user", "content": tool_results})

    return ""

print(run_claude_with_tools("Upgrade customer cust_123 to the pro plan."))
```

> [!warning] Tool results go in `role: "user"` messages
> This is the most common source of confusion when migrating from OpenAI. In Anthropic's API, tool results are sent as a `user` turn with `type: "tool_result"` content blocks — NOT as `role: "tool"` messages.

---

## Streaming

Anthropic uses an event-based stream with strongly-typed events.

```python
def stream_claude(prompt: str) -> str:
    full_text = ""

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text

    print()  # newline
    return full_text

result = stream_claude("Explain the CAP theorem in 3 bullet points.")
```

For streaming with tool use, handle typed events directly:

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    tools=tools,
    messages=[{"role": "user", "content": "Get info for customer cust_456"}]
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            if hasattr(event.content_block, "type"):
                print(f"\n[Block type: {event.content_block.type}]")
        elif event.type == "content_block_delta":
            if hasattr(event.delta, "text"):
                print(event.delta.text, end="", flush=True)
```

---

## Prompt caching

For long system prompts (>1024 tokens for Sonnet), prompt caching reduces cost by 90% on cached tokens.

```python
LONG_SYSTEM = """
You are an expert Python developer with deep knowledge of:
- FastAPI and async patterns
- SQLAlchemy ORM and database optimization
- Redis caching strategies
- Docker and Kubernetes deployment
- Testing with pytest and hypothesis
[... imagine 2000 more tokens of detailed instructions ...]
"""

def cached_query(user_question: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=[
            {
                "type": "text",
                "text": LONG_SYSTEM,
                "cache_control": {"type": "ephemeral"}  # marks this block for caching
            }
        ],
        messages=[{"role": "user", "content": user_question}]
    )
    print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
    print(f"Cache write tokens: {response.usage.cache_creation_input_tokens}")
    return response.content[0].text

# First call: cache_creation_input_tokens > 0 (charged at 1.25× rate)
r1 = cached_query("How do I write a FastAPI dependency for database sessions?")

# Second call: cache_read_input_tokens > 0 (charged at 0.1× rate — 90% discount)
r2 = cached_query("What's the best way to handle connection pooling?")
```

> [!tip] When caching pays off
> Cache write costs 1.25× the normal input rate. Cache reads cost 0.1×. Break-even is after ~2 calls. For a 2000-token system prompt on Claude Sonnet 4.6 ($3/M input tokens):
> - Without cache: $0.006 per call × 100 calls = $0.60
> - With cache (after first call): 2000 × $0.0003 = $0.06 cached read + $0.0075 first write = $0.067 total

---

## Token counting

Count tokens before sending to avoid context limit errors on large documents.

```python
def count_and_check(messages: list[dict], system: str, max_tokens_budget: int = 150_000) -> bool:
    token_count = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        system=system,
        messages=messages
    )
    print(f"Request will use {token_count.input_tokens:,} input tokens")

    if token_count.input_tokens > max_tokens_budget:
        print(f"Warning: exceeds budget of {max_tokens_budget:,} tokens")
        return False
    return True

messages = [{"role": "user", "content": "Summarize this document: " + "word " * 5000}]
count_and_check(messages, "You are a summarizer.", max_tokens_budget=100_000)
```

---

## Model options

| Model | Context | Strengths | Cost (input/output per 1M) |
|-------|---------|-----------|---------------------------|
| `claude-opus-4-7` | 200K | Hardest reasoning, coding, research | $15 / $75 |
| `claude-sonnet-4-6` | 200K | Best capability/cost, general purpose | $3 / $15 |
| `claude-haiku-4-5-20251001` | 200K | Fastest, cheapest, simple tasks | $0.80 / $4 |

> [!tip] Claude Sonnet 4.6 as your default
> For most production use cases, `claude-sonnet-4-6` hits the right balance. Use Haiku for classification, routing, and high-volume extraction where you've already verified quality. Reach for Opus only for tasks where Sonnet measurably falls short.

---

[[03-tool-use-openai]] | [[05-cost-and-rate-limits]]
