# Interview Questions — Function-Calling Data Extractor

---

**Q1: Explain the 4-step function calling cycle. What happens at each step?**

??? "Show answer"
    1. **Send tool definition** — you include a `tools` array in the API call with the function name, description, and JSON schema for its parameters. The model reads this and decides whether to call a function.
    
    2. **Model returns a tool call** — instead of a text reply, the model outputs `tool_calls` with the function name and `arguments` as a JSON string.
    
    3. **Execute the function** — your code parses `json.loads(tool_call.function.arguments)` and calls the actual function. In this project, the "function" is Pydantic validation — we're using tool_use as a structured output mechanism, not to call real tools.
    
    4. **Send results back (optional)** — for agentic workflows, you append the tool result to the message history and send another request. For pure extraction, step 3 is the last step — you return the validated result to the caller.

---

**Q2: Why use `model_json_schema()` instead of writing the JSON schema by hand?**

??? "Show answer"
    `BaseModel.model_json_schema()` generates a valid JSON Schema from the Pydantic model's field definitions, including types, descriptions, and constraints. Writing JSON schemas by hand is error-prone and gets out of sync with your Python models. If you add a field to `InvoiceData`, `model_json_schema()` automatically includes it in the tool definition — no manual sync required.
    
    The descriptions on each field (`Field(..., description="...")`) become the property descriptions in the JSON schema, which the model uses to understand what each field means. Good descriptions directly improve extraction accuracy.

---

**Q3: How do you handle cases where the LLM extracts a field in the wrong type — for example, returning "1,250.00" (a string) for a field typed as `float`?**

??? "Show answer"
    Pydantic V2 coerces types by default: `float` fields will attempt to parse strings like "1250.0". But strings with commas ("1,250.00") or currency symbols ("$1,250") fail coercion.
    
    Three approaches:
    1. **Field description** — add to the schema: `description="Amount as a plain number, no currency symbols or commas"`. The model usually complies.
    2. **Post-processing** — after extraction, normalize amounts with `re.sub(r"[^\d.]", "", str(amount))` before returning.
    3. **Pydantic validator** — use `@field_validator('total_amount', mode='before')` to strip non-numeric characters before Pydantic attempts coercion.

---

**Q4: What's the difference between how OpenAI and Anthropic handle tool call results?**

??? "Show answer"
    OpenAI: tool results are appended as messages with `role: "tool"` and `tool_call_id` matching the original call. The `arguments` field in the tool call is a JSON string requiring `json.loads()`.
    
    Anthropic: tool results are sent as `role: "user"` content blocks with `type: "tool_result"` and `tool_use_id`. The `input` field in the tool_use block is already a Python dict — no `json.loads()` needed.
    
    Both require a second API call if you want the model to respond after seeing the tool result. For pure extraction (where the tool call arguments are the output), no second call is needed with either API.

---

[[05-deployment]]
