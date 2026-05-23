# Interview Questions — Function Calling and Tool Use

---

**Q1: How does function calling differ from JSON mode, and when should you use each?**

??? "Show answer"
    **JSON mode** (`response_format={"type": "json_object"}`) guarantees that the model's output is valid JSON, but does *not* guarantee the structure, field names, or types inside that JSON. The model can return any JSON it decides to generate.

    **Function calling** defines a schema (JSON Schema) and guarantees the model's output conforms to it: correct field names, correct types, enum constraints respected, required fields present. The model's output is a structured tool call, not free-form JSON.

    **Use JSON mode when:**
    - You need valid JSON but the exact schema is flexible or unknown in advance
    - You're collecting open-ended data where structure varies per response
    - Latency is critical and you want to avoid the schema overhead

    **Use function calling when:**
    - The output feeds code that expects specific field names and types
    - You have enum constraints (e.g., `status` must be one of: paid/pending/overdue)
    - You need reliable extraction across thousands of documents
    - You're building an agent that dispatches to real functions

---

**Q2: Walk through the full request-response cycle for a function call that requires executing a tool and returning the result.**

??? "Show answer"
    1. **Define tools:** Pass a list of tool definitions (JSON Schema) in the `tools` parameter
    2. **First API call:** Send user message + tools; model responds with `finish_reason="tool_calls"` and a `tool_calls` list
    3. **Parse call:** Extract `function.name` and `json.loads(function.arguments)` from each tool call
    4. **Execute:** Call the actual function with the parsed arguments
    5. **Build results:** Create one `{"role": "tool", "tool_call_id": ..., "content": json.dumps(result)}` message per tool call
    6. **Second API call:** Send original messages + assistant's tool_calls message + all tool results; model responds with `finish_reason="stop"` and generates the final answer

    Critical constraint: every `tool_call_id` in the assistant's response must have a corresponding `tool` role message. Missing one causes an API error.

---

**Q3: What is the key API difference between OpenAI tool calls and Anthropic tool use?**

??? "Show answer"
    Several differences:

    | Aspect | OpenAI | Anthropic |
    |--------|--------|-----------|
    | Schema key | `parameters` (JSON Schema) | `input_schema` (JSON Schema) |
    | Stop signal | `finish_reason == "tool_calls"` | `stop_reason == "tool_use"` |
    | Arguments format | JSON *string* (must `json.loads()`) | Python *dict* (already parsed) |
    | Tool call location | `message.tool_calls` (list) | `response.content` (mixed blocks) |
    | Result message role | `"tool"` role with `tool_call_id` | `"user"` role with `tool_result` content blocks |
    | Force tool | `tool_choice={"type": "function", "name": "..."}` | `tool_choice={"type": "tool", "name": "..."}` |

    The conceptual loop is identical: define → call → execute → return result. Only the Python API surface differs.

---

**Q4: When should you disable parallel tool calls with `parallel_tool_calls=False`?**

??? "Show answer"
    Disable parallel tool calls when tool outputs have **dependencies** — when tool call B needs to use the result of tool call A.

    Example: Step 1 calls `search_products(query)` → returns product IDs. Step 2 calls `get_product_details(product_id)` using an ID from step 1's result. These must be sequential.

    Also disable when tools have **ordered side effects**: writing to a database, sending emails, or any stateful operation where order matters.

    Keep parallel calls enabled (the default) for independent lookups: weather in multiple cities, stock prices for multiple tickers, extracting entities from multiple documents. These have no dependency and benefit from the reduced latency (all results arrive in one round trip instead of N sequential round trips).

---

**Q5: What does `tool_choice="required"` do, and when would you use it?**

??? "Show answer"
    `tool_choice="required"` forces the model to call at least one tool in every response. It cannot generate a plain text answer — it must produce a tool call.

    **Use it for extraction-only pipelines:** When you're using tool calling purely as a structured extraction mechanism (not for real side effects), you don't want the model to sometimes answer with plain text for "simple" inputs. With `required`, you always get a tool call back — your code can skip the round 2 entirely.

    **Don't use it in conversational agents** where some user messages don't require tool use (e.g., "thank you" or "what did you just say?"). Forcing a tool call on conversational turns produces awkward behavior.

    Anthropic's equivalent: `tool_choice={"type": "any"}` (requires a tool call) or `{"type": "tool", "name": "specific_tool"}` (requires a specific tool).

---

**Q6: How do you use a Pydantic model as a tool schema?**

??? "Show answer"
    Pydantic models can generate JSON Schema automatically via `.model_json_schema()`. This schema can be passed directly as the `parameters` field in an OpenAI tool definition:

    ```python
    from pydantic import BaseModel, Field
    from typing import Optional

    class ExtractedEntity(BaseModel):
        name: str
        entity_type: str
        confidence: float = Field(ge=0.0, le=1.0)
        context: Optional[str] = None

    tool = {
        "type": "function",
        "function": {
            "name": "extract_entity",
            "description": "Extract a named entity from text.",
            "parameters": ExtractedEntity.model_json_schema()
        }
    }
    ```

    After the tool call, parse the arguments back into the Pydantic model for type validation:
    ```python
    raw = json.loads(tool_call.function.arguments)
    entity = ExtractedEntity(**raw)  # Pydantic validates types
    ```

    LangChain's `with_structured_output(MyModel)` does all of this automatically — schema generation, tool call, parsing, and validation — in one line.

---

**Q7: What can go wrong with function calling in production, and how do you handle it?**

??? "Show answer"
    **Common failure modes:**

    1. **Model hallucinates a function name** — requests a function that doesn't exist. Fix: validate `function.name` against your `FUNCTION_MAP` before calling; return an error tool result if not found.

    2. **Malformed arguments** — `json.loads(function.arguments)` raises `JSONDecodeError`. Rare with modern models, but wrap in try/except and retry.

    3. **Type mismatch** — model passes `"49.99"` (string) for a `price: float` field. Fix: use Pydantic validation when parsing; cast where possible.

    4. **Missing required arguments** — model omits a required field. Fix: Pydantic raises `ValidationError`; catch it and retry with a correction message.

    5. **Infinite tool loop** — model keeps calling tools without reaching `finish_reason="stop"`. Fix: add a `max_rounds` counter; break after 5–10 rounds.

    6. **Missing tool result for a tool_call_id** — API error if any tool call is unmatched. Fix: always iterate `choice.message.tool_calls` completely; never break early.

---

[[06-practice-exercises]]
