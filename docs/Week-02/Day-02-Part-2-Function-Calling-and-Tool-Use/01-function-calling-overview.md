# Function Calling Overview

Language models generate text. But most real applications need structured data — not "the price is approximately fifty dollars" but `{"price": 50.00, "currency": "USD"}`. Function calling (also called tool use) is the mechanism that makes LLMs reliable producers of structured output and gives them the ability to invoke external capabilities.

## Learning objectives

- Understand what function calling actually does at the API level
- Distinguish function calling from JSON mode and prompt-based extraction
- Know the full request-response cycle: define → call → execute → respond
- Recognize when tool use is the right approach vs simpler alternatives

---

## The core idea

Without function calling, you parse LLM text hoping it formatted correctly. With function calling, you give the model a schema and it guarantees the output matches it.

```
Without tool use:
  Prompt: "Return JSON with name and price"
  Response: "Here's the JSON: {"name": "Widget", "price": "$49.99"}"
  Problem: must parse, strip text, handle malformed outputs

With tool use:
  Define: extract_product(name: str, price: float)
  Response: tool_call(name="Widget", price=49.99)
  Result: guaranteed valid Python objects
```

---

## The four-step cycle

```python
# Step 1: Define tools (JSON schema)
tools = [{"type": "function", "function": {"name": "...", "parameters": {...}}}]

# Step 2: Send to model with tools
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

# Step 3: Check if model wants to call a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    # Execute the function
    result = execute(tool_call.function.name, json.loads(tool_call.function.arguments))

    # Step 4: Send result back to continue the conversation
    messages.append(response.choices[0].message)  # Assistant's tool call
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
    final_response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
```

---

## Tool use vs JSON mode vs prompting

| Approach | Reliability | Flexibility | Setup |
|----------|-------------|-------------|-------|
| Prompt-based extraction | Low | High | None |
| JSON mode (`response_format`) | Medium | High | Minimal |
| Function calling / tool use | High | Constrained to schema | Schema definition |
| `with_structured_output` (LangChain) | High | Pydantic model | Pydantic class |

**Prompt-based:** "Return JSON with these fields." Works until the model decides to add commentary, nest differently, or omit optional fields. Breaks under distribution shift.

**JSON mode:** Guarantees valid JSON, but the *keys and values* can still be wrong. The model might return `{"sentiment": "somewhat negative"}` when you expected `"negative"`.

**Function calling:** Guarantees the schema — correct field names, correct types, enum constraints respected. The model *cannot* return malformed output.

> [!success] Use tool use for any structured output you'll programmatically consume
> If the output feeds code (not a human), use function calling or `with_structured_output`. The reliability improvement over prompt-based extraction is dramatic, especially for edge cases.

---

## When tool use is the right choice

```python
USE_TOOL_CALLING = [
    "Extracting structured entities from unstructured text (NER++)",
    "Connecting LLMs to external APIs (search, databases, calculators)",
    "Multi-step agent loops where each step has a defined action schema",
    "Any output where you need field-level type guarantees",
    "Replacing fragile regex-based parsing pipelines",
]

DO_NOT_USE_TOOL_CALLING = [
    "Simple yes/no or short text answers",
    "Creative writing or open-ended generation",
    "When the structure isn't known in advance",
    "Situations where partial/incomplete output is acceptable",
]
```

---

## Anatomy of a tool definition

```python
# OpenAI format (also used by LangChain, LiteLLM)
tool = {
    "type": "function",
    "function": {
        "name": "extract_invoice",
        "description": "Extract structured data from an invoice document.",  # Critical: model uses this
        "parameters": {
            "type": "object",
            "properties": {
                "vendor_name": {
                    "type": "string",
                    "description": "Name of the vendor or supplier"
                },
                "total_amount": {
                    "type": "number",
                    "description": "Total invoice amount in USD"
                },
                "invoice_date": {
                    "type": "string",
                    "description": "Invoice date in YYYY-MM-DD format"
                },
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "amount": {"type": "number"}
                        },
                        "required": ["description", "amount"]
                    },
                    "description": "Individual line items on the invoice"
                },
                "status": {
                    "type": "string",
                    "enum": ["paid", "pending", "overdue"],
                    "description": "Current payment status"
                }
            },
            "required": ["vendor_name", "total_amount", "invoice_date"]
        }
    }
}
```

> [!tip] Write good descriptions — they're few-shot prompts for the schema
> The `description` fields in your tool schema guide the model's extraction. "Total invoice amount in USD" is better than "amount" — it tells the model the unit and scope. Treat descriptions as micro-prompts.

---

[[00-agenda]] | [[02-openai-tools]]
