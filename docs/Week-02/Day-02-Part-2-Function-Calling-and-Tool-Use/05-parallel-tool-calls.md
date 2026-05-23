# Parallel Tool Calls

When a model needs multiple pieces of information to answer a question, it can request all of them in one go rather than making sequential round trips. OpenAI and Claude both support parallel tool calls — understanding how to handle them is essential for building efficient agent loops.

## Learning objectives

- Recognize when the model issues parallel tool calls
- Handle multiple `tool_calls` in a single response correctly
- Implement fan-out execution (real parallelism with `asyncio`)
- Know which models support parallel calls and when to disable it
- Debug common parallel tool call errors

---

## What parallel tool calls look like

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"}
                },
                "required": ["from_currency", "to_currency"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo and London, and what's the USD to JPY rate?"}],
    tools=TOOLS,
)

# The model may return MULTIPLE tool calls in one response
print(f"Number of tool calls: {len(response.choices[0].message.tool_calls)}")
# → Number of tool calls: 3

for tc in response.choices[0].message.tool_calls:
    print(f"  {tc.function.name}({tc.function.arguments})")
# → get_weather({"city": "Tokyo"})
# → get_weather({"city": "London"})
# → get_exchange_rate({"from_currency": "USD", "to_currency": "JPY"})
```

---

## Handling parallel calls correctly

The critical rule: **you must add a `tool` result for every `tool_call_id`**. If any are missing, the API returns an error.

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_weather(city: str) -> dict:
    temps = {"Tokyo": 24, "London": 15, "New York": 18, "Sydney": 22}
    return {"city": city, "temperature": temps.get(city, 20), "unit": "celsius"}

def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    rates = {("USD", "JPY"): 149.5, ("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09}
    rate = rates.get((from_currency, to_currency), 1.0)
    return {"from": from_currency, "to": to_currency, "rate": rate}

FUNCTION_MAP = {"get_weather": get_weather, "get_exchange_rate": get_exchange_rate}

def run_parallel_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)  # Add assistant message with all tool_calls

            # Process ALL tool calls — don't break after the first one
            tool_results = []
            for tool_call in choice.message.tool_calls:
                func = FUNCTION_MAP[tool_call.function.name]
                args = json.loads(tool_call.function.arguments)
                result = func(**args)

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,   # Must match the tool_call.id exactly
                    "content": json.dumps(result),
                })

            # Add ALL results before the next API call
            messages.extend(tool_results)

print(run_parallel_tools(
    "What's the weather in Tokyo and London? Also what's USD to JPY rate?"
))
```

---

## Real parallelism with asyncio

The model requests tools in parallel, but your code executes them sequentially by default. For I/O-bound functions (API calls, database queries), use asyncio:

```python
import os
import json
import asyncio
import httpx
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def fetch_weather_async(city: str) -> dict:
    """Simulate an async weather API call."""
    await asyncio.sleep(0.1)  # Simulates network I/O
    return {"city": city, "temperature": 20, "condition": "clear"}

async def fetch_stock_price_async(ticker: str) -> dict:
    """Simulate an async stock price API call."""
    await asyncio.sleep(0.15)  # Simulates network I/O
    prices = {"AAPL": 213.50, "GOOGL": 175.20, "MSFT": 415.80}
    return {"ticker": ticker, "price": prices.get(ticker, 100.0), "currency": "USD"}

ASYNC_FUNCTION_MAP = {
    "fetch_weather": fetch_weather_async,
    "fetch_stock_price": fetch_stock_price_async,
}

ASYNC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_weather",
            "description": "Fetch current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_stock_price",
            "description": "Fetch current stock price for a ticker symbol.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
        }
    }
]

async def run_with_real_parallelism(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = await aclient.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=ASYNC_TOOLS,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            # Execute ALL tool calls concurrently
            async def execute_one(tool_call):
                func = ASYNC_FUNCTION_MAP[tool_call.function.name]
                args = json.loads(tool_call.function.arguments)
                result = await func(**args)
                return {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }

            tool_results = await asyncio.gather(
                *[execute_one(tc) for tc in choice.message.tool_calls]
            )
            messages.extend(tool_results)

import asyncio
result = asyncio.run(run_with_real_parallelism(
    "What's the weather in Tokyo and what's the AAPL stock price?"
))
print(result)
```

---

## Disabling parallel calls

Sometimes sequential is correct — when tool call 2 depends on the output of tool call 1:

```python
# Disable parallel tool calls (OpenAI)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    parallel_tool_calls=False,  # Model will call tools one at a time
)
```

**When to disable:**
- Tool B uses the result of Tool A (dependency chain)
- Tools have side effects that must happen in order (database writes)
- Debugging: sequential is easier to trace

**When parallel is fine (the common case):**
- Independent lookups: weather + stock + currency all at once
- Multiple entity extractions from the same text
- Fan-out search: query multiple indexes simultaneously

> [!warning] Always handle the multi-call case even if you expect one
> Even with `parallel_tool_calls=True` (default), some queries trigger only one tool call. Your loop must handle `len(tool_calls) == 1` and `len(tool_calls) > 1` — don't assume.

---

## Parallel tool calls in Anthropic Claude

Claude also supports parallel tool use — multiple `tool_use` blocks in a single response:

```python
import anthropic
import json
import os

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    tools=tools,  # Same tool definitions as above
    messages=[{"role": "user", "content": "Weather in Tokyo and London?"}]
)

# Multiple tool_use blocks in response.content
tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
print(f"Claude requested {len(tool_use_blocks)} parallel calls")

# Execute all, collect results
results = []
for block in tool_use_blocks:
    result = FUNCTION_MAP[block.name](**block.input)
    results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": json.dumps(result),
    })

# Send all results in one user message
messages = [
    {"role": "user", "content": "Weather in Tokyo and London?"},
    {"role": "assistant", "content": response.content},
    {"role": "user", "content": results},  # List of tool_result blocks
]
```

---

[[04-structured-extraction]] | [[06-practice-exercises]]
