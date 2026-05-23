# OpenAI Tools API

OpenAI's tools API lets you define functions the model can call, inspect the call it wants to make, execute the function yourself, and feed the result back into the conversation. The model never actually executes code — it generates a structured request and you run it.

## Learning objectives

- Define tools with JSON Schema and pass them to the API
- Parse `tool_calls` from the response and extract arguments
- Implement the full two-round conversation pattern
- Use `tool_choice` to force or restrict tool calls
- Build a multi-tool dispatcher

---

## Minimal working example

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                },
                "required": ["city"]
            }
        }
    }
]

# Simulated function
def get_weather(city: str, unit: str = "celsius") -> dict:
    return {"city": city, "temperature": 22, "unit": unit, "condition": "sunny"}

# Round 1: model decides to call the tool
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools
)

print(f"Finish reason: {response.choices[0].finish_reason}")
# → tool_calls

tool_call = response.choices[0].message.tool_calls[0]
print(f"Function: {tool_call.function.name}")
print(f"Arguments: {tool_call.function.arguments}")
# → {"city": "Tokyo"}
```

---

## The full two-round pattern

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search product catalog by name or category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"},
                    "max_price": {"type": "number", "description": "Maximum price in USD"},
                    "in_stock": {"type": "boolean", "description": "Filter to in-stock items only"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get full details for a product by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Unique product identifier"}
                },
                "required": ["product_id"]
            }
        }
    }
]

# Simulated function implementations
def search_products(query: str, max_price: float = None, in_stock: bool = None) -> list:
    results = [
        {"id": "P001", "name": "Wireless Headphones", "price": 79.99, "in_stock": True},
        {"id": "P002", "name": "Noise-Cancelling Headphones", "price": 149.99, "in_stock": False},
        {"id": "P003", "name": "Sport Earbuds", "price": 49.99, "in_stock": True},
    ]
    if max_price:
        results = [r for r in results if r["price"] <= max_price]
    if in_stock is not None:
        results = [r for r in results if r["in_stock"] == in_stock]
    return results

def get_product_details(product_id: str) -> dict:
    catalog = {
        "P001": {"id": "P001", "name": "Wireless Headphones", "price": 79.99,
                 "description": "30hr battery, Bluetooth 5.2", "rating": 4.5}
    }
    return catalog.get(product_id, {"error": "Product not found"})

FUNCTION_MAP = {
    "search_products": search_products,
    "get_product_details": get_product_details,
}

def chat_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            # Add assistant message (with tool_calls) to history
            messages.append(choice.message)

            # Execute each tool call
            for tool_call in choice.message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"  → Calling {func_name}({func_args})")
                result = FUNCTION_MAP[func_name](**func_args)

                # Add tool result to history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

# Test
print(chat_with_tools("Show me wireless headphones under $100 that are in stock"))
print()
print(chat_with_tools("Get details for product P001"))
```

---

## Controlling tool use with `tool_choice`

```python
# Default: model decides when to call tools
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools
)

# Force the model to always call a specific tool
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_entities"}}
)

# Force the model to call any tool (never generate text directly)
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools,
    tool_choice="required"
)

# Prevent tool calls entirely
response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools,
    tool_choice="none"
)
```

> [!tip] Use `tool_choice="required"` for extraction tasks
> When you're using tool calling purely for structured extraction (not for side effects), force the model to always call the tool. Otherwise, for very simple inputs, the model may answer with plain text instead.

---

## Extraction-only pattern (no round 2 needed)

For pure extraction, you don't need to send the result back to the model:

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_contact",
        "description": "Extract contact information from text.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "company": {"type": "string"},
            },
            "required": ["name"]
        }
    }
}

def extract_contact(text: str) -> dict | None:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract contact information from the text."},
            {"role": "user", "content": text}
        ],
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_contact"}},
    )
    tool_call = response.choices[0].message.tool_calls
    if not tool_call:
        return None
    return json.loads(tool_call[0].function.arguments)

# Test
texts = [
    "Hi, I'm Sarah Chen from Acme Corp. Reach me at sarah.chen@acme.com or 415-555-0199.",
    "Contact John Smith for pricing: jsmith@startup.io",
    "No contact info here, just a regular sentence.",
]
for t in texts:
    result = extract_contact(t)
    print(f"Input: {t[:60]}")
    print(f"Extracted: {result}\n")
```

---

## Error handling

```python
def safe_tool_call(messages: list, tools: list, function_map: dict) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools
        )
    except Exception as e:
        return f"API error: {e}"

    choice = response.choices[0]

    if choice.finish_reason == "stop":
        return choice.message.content

    if choice.finish_reason == "tool_calls":
        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            if func_name not in function_map:
                # Model hallucinated a function name — return error
                return f"Unknown function requested: {func_name}"

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                return "Model returned malformed arguments"

            try:
                result = function_map[func_name](**args)
            except TypeError as e:
                return f"Function call failed: {e}"

    return "Unexpected finish reason"
```

> [!warning] Always validate function arguments before executing
> The model can occasionally hallucinate function names or pass arguments with wrong types. Wrap function dispatch in try/except and validate required parameters exist before calling.

---

[[01-function-calling-overview]] | [[03-anthropic-tool-use]]
