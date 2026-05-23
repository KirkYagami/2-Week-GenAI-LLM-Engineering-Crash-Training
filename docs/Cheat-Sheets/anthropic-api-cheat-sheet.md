# Anthropic API Cheat Sheet

Quick reference for the Anthropic Python SDK.

---

## Basic message

```python
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "What is RAG?"}],
)
text = message.content[0].text
tokens_in = message.usage.input_tokens
tokens_out = message.usage.output_tokens
```

## Async

```python
import anthropic
aclient = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = await aclient.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Streaming

```python
with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    messages=[{"role": "user", "content": "Count to 10."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
final = stream.get_final_message()
```

## Tool use (function calling)

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}]

# Step 1: Initial call
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=500,
    tools=tools,
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
)

# Step 2: Execute the tool
for block in response.content:
    if block.type == "tool_use":
        # block.input is already a dict — no json.loads() needed
        city = block.input["city"]
        result = get_weather(city)

        # Step 3: Return result
        response2 = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            tools=tools,
            messages=[
                {"role": "user", "content": "Weather in Tokyo?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }],
                },
            ],
        )
```

## Multi-turn conversation

```python
messages = []
while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system="You are a helpful assistant.",
        messages=messages,
    )
    reply = response.content[0].text
    messages.append({"role": "assistant", "content": reply})
    print(f"Claude: {reply}")
```

## Models quick reference

| Model | Context | Input $/1M | Output $/1M | Best for |
|-------|---------|------------|-------------|----------|
| `claude-opus-4-7` | 200k | $15.00 | $75.00 | Complex reasoning, agents |
| `claude-sonnet-4-6` | 200k | $3.00 | $15.00 | Balanced capability |
| `claude-haiku-4-5-20251001` | 200k | $0.80 | $4.00 | Fast, cost-effective |

## Key differences from OpenAI API

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System prompt | `messages[0].role="system"` | Separate `system=` param |
| Tool arguments | JSON string (requires `json.loads`) | Python dict (already parsed) |
| Tool results | `role="tool"` message | `role="user"` with `tool_result` block |
| Streaming | `AsyncStream` | Context manager `.stream()` |
| Content | `choices[0].message.content` (str) | `content[0].text` (list of blocks) |
