# Anthropic Tool Use

Anthropic's Claude supports tool use with a slightly different API shape than OpenAI's, but the conceptual pattern is identical: define tools → model requests a call → you execute → you return the result. If you understand OpenAI's tools, this is a 10-minute translation exercise.

## Learning objectives

- Define tools in Anthropic's format (`input_schema` vs OpenAI's `parameters`)
- Parse `tool_use` content blocks from Claude's response
- Implement the multi-turn tool loop with `tool_result` blocks
- Use `tool_choice` to control when Claude calls tools
- Know the key API differences between OpenAI and Anthropic

---

## Minimal working example

```python
import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {           # Anthropic uses input_schema, not parameters
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

def get_weather(city: str, unit: str = "celsius") -> dict:
    return {"city": city, "temperature": 22, "unit": unit, "condition": "sunny"}

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)

print(f"Stop reason: {response.stop_reason}")  # → tool_use
print(f"Content: {response.content}")
# [ToolUseBlock(id='toolu_...', input={'city': 'Paris'}, name='get_weather', type='tool_use')]
```

---

## API shape comparison

| Concept | OpenAI | Anthropic |
|---------|--------|-----------|
| Tool list key | `tools` | `tools` |
| Schema key | `parameters` | `input_schema` |
| Stop signal | `finish_reason == "tool_calls"` | `stop_reason == "tool_use"` |
| Tool call location | `message.tool_calls` | `response.content` (list of blocks) |
| Tool call type | `ChatCompletionMessageToolCall` | `ToolUseBlock` |
| Arguments key | `function.arguments` (JSON string) | `input` (dict — already parsed) |
| Result message role | `"tool"` | `"user"` with `tool_result` block |

> [!tip] Anthropic's `input` is already a dict
> Unlike OpenAI where `tool_call.function.arguments` is a JSON string you must parse, Anthropic's `tool_use_block.input` is already a Python dict. Skip the `json.loads()` call.

---

## Full multi-turn tool loop

```python
import os
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "search_database",
        "description": "Search the product database for items matching a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "category": {
                    "type": "string",
                    "enum": ["electronics", "clothing", "books", "home"],
                    "description": "Product category filter"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_discount",
        "description": "Calculate the final price after applying a discount code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "original_price": {"type": "number"},
                "discount_code": {"type": "string"}
            },
            "required": ["original_price", "discount_code"]
        }
    }
]

def search_database(query: str, category: str = None) -> list:
    results = [
        {"id": "E001", "name": "Bluetooth Speaker", "price": 59.99, "category": "electronics"},
        {"id": "E002", "name": "Smart Watch", "price": 199.99, "category": "electronics"},
        {"id": "B001", "name": "Python Cookbook", "price": 34.99, "category": "books"},
    ]
    filtered = [r for r in results if query.lower() in r["name"].lower()]
    if category:
        filtered = [r for r in filtered if r["category"] == category]
    return filtered

def calculate_discount(original_price: float, discount_code: str) -> dict:
    codes = {"SAVE10": 0.10, "SAVE20": 0.20, "HALFOFF": 0.50}
    discount = codes.get(discount_code.upper(), 0)
    final_price = original_price * (1 - discount)
    return {
        "original": original_price,
        "discount_pct": discount * 100,
        "final_price": round(final_price, 2),
        "valid_code": discount > 0
    }

FUNCTION_MAP = {
    "search_database": search_database,
    "calculate_discount": calculate_discount,
}

def chat_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Extract text from content blocks
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return " ".join(text_blocks)

        if response.stop_reason == "tool_use":
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response.content})

            # Process all tool_use blocks
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                func_name = block.name
                func_args = block.input  # Already a dict — no json.loads needed

                print(f"  → Calling {func_name}({func_args})")
                result = FUNCTION_MAP[func_name](**func_args)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            # Add tool results as user message
            messages.append({"role": "user", "content": tool_results})

# Test
import json
print(chat_with_tools("Find me a Bluetooth speaker and apply discount code SAVE20"))
```

---

## Controlling tool use

```python
# Default: Claude decides when to call tools
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=messages,
)

# Force Claude to call a specific tool
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_entities"},
    messages=messages,
)

# Force Claude to call any tool
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "any"},
    messages=messages,
)

# Prevent tool calls
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "none"},  # Note: Anthropic uses dict, not string
    messages=messages,
)
```

---

## Extraction-only pattern

```python
import os
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

EXTRACT_TOOL = {
    "name": "extract_job_posting",
    "description": "Extract structured data from a job posting.",
    "input_schema": {
        "type": "object",
        "properties": {
            "job_title": {"type": "string"},
            "company": {"type": "string"},
            "location": {"type": "string"},
            "salary_min": {"type": "number", "description": "Minimum salary in USD annually"},
            "salary_max": {"type": "number", "description": "Maximum salary in USD annually"},
            "remote": {"type": "boolean", "description": "Whether remote work is offered"},
            "required_skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of required technical skills"
            }
        },
        "required": ["job_title", "company"]
    }
}

def extract_job(text: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_job_posting"},
        messages=[{"role": "user", "content": f"Extract job info from:\n\n{text}"}]
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input  # Already a dict
    return {}

# Test
posting = """
Senior ML Engineer at TechCorp — San Francisco (Remote OK)
Salary: $180,000–$240,000/year
We need 5+ years Python, experience with PyTorch or TensorFlow, and strong MLOps skills.
"""
result = extract_job(posting)
print(result)
# {'job_title': 'Senior ML Engineer', 'company': 'TechCorp', 'location': 'San Francisco',
#  'salary_min': 180000, 'salary_max': 240000, 'remote': True, 'required_skills': ['Python', 'PyTorch', 'TensorFlow', 'MLOps']}
```

> [!warning] Handle `tool_use` blocks in streaming responses
> If you use `client.messages.stream()`, tool_use blocks arrive as `content_block_start` + `content_block_delta` + `content_block_stop` events. The `input` dict is built incrementally — use `block.input` only after `content_block_stop` for that block.

---

[[02-openai-tools]] | [[04-structured-extraction]]
