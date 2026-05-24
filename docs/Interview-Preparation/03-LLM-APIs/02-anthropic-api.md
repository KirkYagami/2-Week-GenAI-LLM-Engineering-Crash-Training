# Anthropic API

Knowing the Anthropic API signals you've worked with more than one provider and can reason about API design tradeoffs — a signal interviewers at companies that use Claude will explicitly look for.

---

## Q1: What are the key differences between the OpenAI and Anthropic APIs?

??? "Show answer"
    | | OpenAI | Anthropic |
    |-|--------|-----------|
    | **System prompt** | `{"role": "system", "content": "..."}` in messages | Separate `system` parameter |
    | **Streaming** | Iterator of chunks | Context manager + event stream |
    | **Tool results** | `{"role": "tool", "tool_call_id": "..."}` | `{"role": "user", "content": [{"type": "tool_result", ...}]}` |
    | **Tool arguments** | String — must `json.loads()` | Already a dict — no parsing needed |
    | **Max tokens** | `max_tokens` (optional, defaults vary) | `max_tokens` required |
    | **Model naming** | `gpt-4o`, `gpt-4o-mini` | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` |

    The most common mistake when switching from OpenAI to Anthropic: trying to `json.loads()` tool arguments. Anthropic already parses them — `block.input` is a dict, not a string.

---

## Q2: Write a basic Anthropic API call with a system prompt.

??? "Show answer"
    ```python
    import os
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a concise technical writer. Reply in plain text only.",
        messages=[
            {"role": "user", "content": "Explain what a transformer is in two sentences."}
        ],
    )

    print(response.content[0].text)
    ```

    Key difference from OpenAI: `max_tokens` is **required** in Anthropic. The response content is a list — always access `response.content[0].text` for a standard text response.

---

## Q3: How does Anthropic's streaming API work?

??? "Show answer"
    Anthropic streaming uses a context manager and yields typed events rather than raw chunks.

    ```python
    import os
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write a haiku about embeddings."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    final = stream.get_final_message()
    print(f"\nInput tokens: {final.usage.input_tokens}")
    print(f"Output tokens: {final.usage.output_tokens}")
    ```

    The context manager ensures the stream is properly closed. `stream.text_stream` yields text fragments only — it filters out non-text events. `get_final_message()` gives you the complete message with usage stats after the stream ends.

---

## Q4: How does tool use work in the Anthropic API?

??? "Show answer"
    ```python
    import os
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        }
    }]

    messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    # Check for tool use
    for block in response.content:
        if block.type == "tool_use":
            # block.input is ALREADY a dict — no json.loads() needed
            city = block.input["city"]
            weather_result = f"68°F, cloudy in {city}"

            # Append the assistant response and tool result
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": weather_result,
                }]
            })

    # Final call
    final = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, tools=tools, messages=messages)
    print(final.content[0].text)
    ```

    The tool result goes back as a `user` message with `type: "tool_result"` — not a separate role like OpenAI's `"tool"` role.

---

## Q5: Which Anthropic model should you use for which tasks?

??? "Show answer"
    | Model | Strengths | When to use |
    |-------|-----------|-------------|
    | `claude-opus-4-7` | Most capable, best reasoning | Complex analysis, nuanced writing, hard coding tasks |
    | `claude-sonnet-4-6` | Best balance of capability and speed | Most production workloads — RAG, agents, function calling |
    | `claude-haiku-4-5` | Fastest and cheapest | Classification, routing, simple extraction, high-volume tasks |

    In a RAG system: use `claude-haiku-4-5` for query analysis and routing, `claude-sonnet-4-6` for answer generation. This splits cost while preserving quality at the generation stage.

    Anthropic also supports **prompt caching** — if your system prompt is large (> 1024 tokens), you can cache it and pay only for read tokens on subsequent calls. Cache TTL is 5 minutes by default.

---

*Previous: [OpenAI API](01-openai-api.md) | Next: [Embeddings](03-embeddings.md)*
