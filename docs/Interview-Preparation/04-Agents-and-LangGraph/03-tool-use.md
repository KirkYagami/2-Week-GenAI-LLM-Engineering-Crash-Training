# Tool Use

Tool use / function calling is the primary mechanism for connecting LLMs to the real world. Interviewers test this because it's where most production systems actually break.

---

## Q1: Walk me through the complete tool use cycle in the OpenAI API.

??? "Show answer"
    Four steps — the model never executes tools itself; your code does:

    **Step 1**: Send request with tool definitions
    **Step 2**: Model returns a response with `tool_calls` instead of `content`
    **Step 3**: Your code executes each tool call and collects results
    **Step 4**: Append assistant message + tool results, call the API again for the final answer

    ```python
    import json, os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def search_docs(query: str) -> str:
        return f"Found 3 results for '{query}': [result1, result2, result3]"

    tools = [{"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the documentation",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }}]

    messages = [{"role": "user", "content": "How do I reset my password?"}]

    # Step 1 & 2: Initial call
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
    msg = response.choices[0].message

    # Step 3 & 4: Execute and return
    if msg.tool_calls:
        messages.append(msg)  # MUST append the assistant message first
        for tc in msg.tool_calls:
            result = search_docs(**json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        final = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
        print(final.choices[0].message.content)
    ```

    The most common bug: forgetting to append the assistant message before appending tool results. The API validates message ordering strictly.

---

## Q2: What's the difference between `tool_choice: "auto"`, `"required"`, and a specific function?

??? "Show answer"
    | `tool_choice` | Behaviour |
    |---------------|-----------|
    | `"auto"` (default) | Model decides whether to call a tool or respond directly |
    | `"none"` | Model cannot call any tools — respond with text only |
    | `"required"` | Model must call at least one tool |
    | `{"type": "function", "function": {"name": "..."}}` | Model must call this specific function |

    Use `"required"` when you need the model to always return structured data (extraction pipelines). Use a specific function when you need deterministic tool selection — useful for extraction where you always want the same schema filled.

    ```python
    # Force the model to always extract using this schema
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_invoice"}},
    )
    ```

---

## Q3: How do you handle tool execution errors and feed them back to the model?

??? "Show answer"
    Return error information as the tool result — the model can then reason about the failure and either try again, use a different tool, or explain the issue to the user.

    ```python
    import json, os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def risky_tool(query: str) -> str:
        try:
            return expensive_database_call(query)
        except TimeoutError:
            return "ERROR: Database query timed out after 5 seconds"
        except ValueError as e:
            return f"ERROR: Invalid query format — {e}"

    # In your tool execution loop:
    for tc in msg.tool_calls:
        result = risky_tool(**json.loads(tc.function.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,   # error string is valid — model will handle it
        })
    ```

    Never raise an unhandled exception during tool execution — the loop breaks and you lose the conversation state. Always return a string result, even if that string is an error message.

---

## Q4: What is parallel tool calling and how do you handle it?

??? "Show answer"
    When a request could benefit from multiple independent tool calls, the model returns multiple `tool_calls` in a single response. You execute them all and return all results before the next API call.

    ```python
    import json, asyncio, os
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def execute_tool(name: str, args: dict) -> str:
        if name == "search":
            return await async_search(args["query"])
        elif name == "get_weather":
            return await async_weather(args["city"])
        return f"Unknown tool: {name}"

    async def run_with_parallel_tools(messages: list) -> str:
        response = await aclient.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)

        # Execute all tool calls in parallel
        results = await asyncio.gather(*[
            execute_tool(tc.function.name, json.loads(tc.function.arguments))
            for tc in msg.tool_calls
        ])

        for tc, result in zip(msg.tool_calls, results):
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        final = await aclient.chat.completions.create(model="gpt-4o", messages=messages)
        return final.choices[0].message.content
    ```

    Always append tool results in the same order as the `tool_calls` list — the model matches them by `tool_call_id`.

---

## Q5: How do you design tool schemas for reliability?

??? "Show answer"
    Tool schema quality directly affects how reliably the model calls your tools. Guidelines:

    1. **Descriptions are prompts** — the `description` field is model-readable. Be specific: "Search the product documentation for answers about billing, plans, and account management" beats "Search docs."

    2. **Use enums for constrained inputs** — if only specific values are valid, enumerate them in the schema so the model can't hallucinate an invalid value.

    3. **Keep tool count small** — every additional tool is another surface for the model to misuse. Remove tools that overlap in purpose.

    4. **Name functions unambiguously** — `search_billing_docs` is better than `search` when there are multiple search tools.

    ```python
    tool = {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a new support ticket. Use ONLY after collecting the customer's issue description and contact email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_description": {
                        "type": "string",
                        "description": "Verbatim description of the customer's issue, at least 20 characters"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Priority based on business impact"
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer's email address for follow-up"
                    }
                },
                "required": ["issue_description", "priority", "customer_email"],
                "additionalProperties": False,
            }
        }
    }
    ```

---

*Previous: [LangGraph](02-langgraph.md) | Next: [LoRA and QLoRA](../05-Fine-Tuning/01-lora-qlora.md)*
