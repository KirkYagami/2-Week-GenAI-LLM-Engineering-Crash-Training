# Tool Use with OpenAI

Tool use (function calling) lets the model decide to call external functions during a response, then use the results to complete its answer. It's how you connect an LLM to live data, APIs, databases, and code execution.

## Learning objectives

- Define tools using the OpenAI function schema
- Handle tool call responses and inject results back into the conversation
- Execute multiple tools in parallel using `parallel_tool_calls`
- Use `strict: true` for guaranteed schema adherence

---

## The tool use loop

Tool use is not a single API call — it's a conversation loop:

```
1. You send a message with tool definitions
2. Model responds with tool_calls (not text yet)
3. You execute the tool(s)
4. You send tool results back as role="tool" messages
5. Model responds with final text answer
```

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location. Use this when the user asks about weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

# Step 2: Fake implementation (replace with real API call)
def get_weather(location: str, unit: str = "celsius") -> dict:
    return {"location": location, "temperature": 22, "unit": unit, "condition": "Sunny"}

# Step 3: Full tool use loop
def run_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # Loop until no more tool calls
    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        messages.append(assistant_message)  # include assistant's tool_calls

        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "get_weather":
                result = get_weather(**fn_args)
            else:
                result = {"error": f"Unknown function: {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

    return response.choices[0].message.content

print(run_with_tools("What's the weather like in Tokyo?"))
# Output: The weather in Tokyo is currently sunny at 22°C.
```

---

## Strict mode

With `"strict": True`, the model is guaranteed to return valid JSON that exactly matches your schema — no extra fields, no missing required fields.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_contact",
            "description": "Extract contact information from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": ["string", "null"]}
                },
                "required": ["name", "email", "phone"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Extract contact info: John Smith, john@example.com, no phone listed."
    }],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_contact"}}
)

args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
print(args)
# Output: {'name': 'John Smith', 'email': 'john@example.com', 'phone': None}
```

> [!tip] `tool_choice` for forced extraction
> Setting `tool_choice={"type": "function", "function": {"name": "..."}}` forces the model to call that specific function. Useful for extraction tasks where you always want structured output, not a text response.

---

## Parallel tool calls

When multiple independent tools are needed, the model can call them all at once.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"}
                },
                "required": ["ticker"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get recent news headlines for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"}
                },
                "required": ["company"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

# Fake implementations
def get_stock_price(ticker: str) -> dict:
    prices = {"AAPL": 189.50, "MSFT": 412.30, "GOOGL": 175.20}
    return {"ticker": ticker, "price": prices.get(ticker, 0), "currency": "USD"}

def get_company_news(company: str) -> dict:
    return {"company": company, "headlines": [f"{company} reports strong earnings", f"{company} announces new product"]}

TOOL_REGISTRY = {
    "get_stock_price": get_stock_price,
    "get_company_news": get_company_news
}

def run_parallel_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        parallel_tool_calls=True  # default True, shown explicitly
    )

    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # Execute all tool calls (could run truly parallel with asyncio)
        for tool_call in assistant_message.tool_calls:
            fn = TOOL_REGISTRY[tool_call.function.name]
            args = json.loads(tool_call.function.arguments)
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )

    return response.choices[0].message.content

print(run_parallel_tools("Give me a quick overview of Apple (AAPL) — price and recent news."))
```

---

## Building a multi-tool assistant

A realistic assistant that uses tools to answer questions about an internal database.

```python
from datetime import datetime

# Tool implementations
def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    return {
        "query": query,
        "results": [
            {"id": "doc-1", "content": f"Relevant content for '{query}'", "score": 0.92},
            {"id": "doc-2", "content": f"Another relevant passage", "score": 0.87}
        ][:top_k]
    }

def get_current_time() -> dict:
    return {"datetime": datetime.now().isoformat(), "timezone": "UTC"}

def create_ticket(title: str, description: str, priority: str) -> dict:
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {"ticket_id": ticket_id, "status": "created", "title": title, "priority": priority}

SUPPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the support knowledge base for relevant articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": False  # top_k is optional
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket when the user's issue cannot be resolved immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}
                },
                "required": ["title", "description", "priority"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

SUPPORT_REGISTRY = {
    "search_knowledge_base": search_knowledge_base,
    "create_ticket": create_ticket,
    "get_current_time": get_current_time
}

SYSTEM_PROMPT = """You are a support agent. Use search_knowledge_base first to find relevant articles.
Only create a ticket if the knowledge base doesn't resolve the issue. Always be concise."""

def support_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=SUPPORT_TOOLS
    )

    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        for tc in assistant_message.tool_calls:
            fn = SUPPORT_REGISTRY[tc.function.name]
            args = json.loads(tc.function.arguments)
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=SUPPORT_TOOLS
        )

    return response.choices[0].message.content

print(support_agent("My login isn't working. I've tried resetting my password twice."))
```

---

## Common mistakes

> [!warning] Always append `assistant_message` before tool results
> When the model returns `tool_calls`, you must include the full assistant message (with its `tool_calls` field) in your next request. If you only send `tool` role messages without the preceding assistant message, you get a 400 error.

> [!warning] `strict: True` requires `additionalProperties: False`
> In strict mode, every property in nested objects must also have `additionalProperties: false` and all properties must be listed in `required` (use `null` type for optional fields). Forgetting this causes the API to silently fall back to non-strict mode.

> [!warning] Don't block on tool execution in production
> If a tool call hits an external API that takes 3s, and the model made 5 parallel tool calls, sequential execution takes 15s. Use `asyncio.gather()` to execute parallel tool calls concurrently.

---

[[02-vision-and-multimodal]] | [[04-anthropic-messages-api]]
