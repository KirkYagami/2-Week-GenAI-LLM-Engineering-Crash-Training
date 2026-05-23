# Tool-Calling Agents

Tool-calling agents use the native function calling API rather than text-parsed ReAct format. The model doesn't write "Action: search[...]" — it issues a structured `tool_calls` request that your code executes and feeds back. This is faster, more reliable, and the approach used in most production agents today.

## Learning objectives

- Build a complete tool-calling agent with the OpenAI API
- Implement tool registration with type-safe wrappers
- Add agent-level safeguards: step limits, token budgeting, error recovery
- Build a domain-specific research agent end-to-end
- Test agent behavior systematically

---

## Tool registration pattern

```python
import os
import json
import inspect
from typing import Callable, get_type_hints
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def tool(description: str, **param_descriptions: str):
    """Decorator to register a function as an agent tool."""
    def decorator(func: Callable) -> Callable:
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        params = {}

        TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}

        for name, param in sig.parameters.items():
            if name == "return":
                continue
            python_type = hints.get(name, str)
            params[name] = {
                "type": TYPE_MAP.get(python_type, "string"),
                "description": param_descriptions.get(name, f"The {name} parameter"),
            }
            if param.default is inspect.Parameter.empty:
                func.__required__ = getattr(func, "__required__", []) + [name]

        func.__tool_def__ = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": getattr(func, "__required__", []),
                },
            }
        }
        return func
    return decorator

# Register tools with the decorator
@tool("Search for information on a topic.", query="The search query")
def web_search(query: str) -> str:
    results = {
        "climate change": "Global temperatures have risen 1.1°C since pre-industrial times.",
        "renewable energy": "Solar and wind now account for over 30% of global electricity.",
        "electric vehicles": "EV sales reached 14 million units globally in 2023.",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"General result for: {query}"

@tool("Get the current price of a stock.", ticker="Stock ticker symbol like AAPL or GOOGL")
def get_stock_price(ticker: str) -> dict:
    prices = {"AAPL": 213.50, "GOOGL": 175.20, "TSLA": 250.80, "NVDA": 875.40}
    price = prices.get(ticker.upper())
    return {"ticker": ticker.upper(), "price": price, "found": price is not None}

@tool(
    "Calculate a mathematical expression.",
    expression="A math expression like '100 * 0.15' or '(10 + 5) / 3'"
)
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{result}"
    except Exception as e:
        return f"Error: {e}"

# Build tool registry
REGISTERED_TOOLS = [web_search, get_stock_price, calculate]
TOOL_DEFS = [f.__tool_def__ for f in REGISTERED_TOOLS]
FUNCTION_MAP = {f.__name__: f for f in REGISTERED_TOOLS}
```

---

## The complete tool-calling agent

```python
import os
import json
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class AgentRun:
    question: str
    answer: str = ""
    steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    error: str = ""

def run_agent(
    question: str,
    tools: list[dict],
    function_map: dict,
    system_prompt: str,
    max_steps: int = 10,
    max_output_tokens: int = 500,
) -> AgentRun:
    run = AgentRun(question=question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    while run.steps < max_steps:
        run.steps += 1

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                max_tokens=max_output_tokens,
            )
        except Exception as e:
            run.error = f"API error: {e}"
            return run

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            run.answer = choice.message.content
            return run

        if choice.finish_reason == "length":
            run.error = "Max tokens reached mid-response"
            return run

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                func_name = tc.function.name
                run.tool_calls.append({"step": run.steps, "tool": func_name, "args": tc.function.arguments})

                if func_name not in function_map:
                    result = {"error": f"Tool '{func_name}' not found. Available: {list(function_map.keys())}"}
                else:
                    try:
                        args = json.loads(tc.function.arguments)
                        result = function_map[func_name](**args)
                    except (json.JSONDecodeError, TypeError) as e:
                        result = {"error": f"Failed to call {func_name}: {e}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

    run.error = f"Max steps ({max_steps}) reached"
    return run

# Test the agent
SYSTEM = """You are a financial research assistant. Use your tools to answer questions accurately.
Always search for information before answering factual questions.
Always use the calculator for mathematical computations."""

questions = [
    "What's the current price of NVDA and AAPL, and what's the total value if I own 10 shares of each?",
    "What percentage of global electricity comes from renewable energy?",
]

for q in questions:
    print(f"\nQ: {q}")
    result = run_agent(q, TOOL_DEFS, FUNCTION_MAP, SYSTEM, max_steps=8)
    print(f"Steps: {result.steps} | Tools used: {[tc['tool'] for tc in result.tool_calls]}")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"A: {result.answer}")
```

---

## Agent testing patterns

```python
from dataclasses import dataclass

@dataclass
class AgentTestCase:
    question: str
    expected_tools: list[str]        # Tools that should be called
    expected_keywords: list[str]     # Keywords that should appear in the answer
    max_allowed_steps: int = 8

def test_agent(test_cases: list[AgentTestCase]) -> None:
    passed = 0
    for tc in test_cases:
        result = run_agent(tc.question, TOOL_DEFS, FUNCTION_MAP, SYSTEM)
        tools_used = {call["tool"] for call in result.tool_calls}

        tool_ok = all(t in tools_used for t in tc.expected_tools)
        keyword_ok = all(k.lower() in result.answer.lower() for k in tc.expected_keywords)
        steps_ok = result.steps <= tc.max_allowed_steps

        status = "PASS" if (tool_ok and keyword_ok and steps_ok and not result.error) else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"{status} | Q: {tc.question[:50]}")
        if status == "FAIL":
            if not tool_ok:
                print(f"       Missing tools: {set(tc.expected_tools) - tools_used}")
            if not keyword_ok:
                missing = [k for k in tc.expected_keywords if k.lower() not in result.answer.lower()]
                print(f"       Missing keywords: {missing}")

    print(f"\n{passed}/{len(test_cases)} tests passed")

TEST_SUITE = [
    AgentTestCase(
        question="What is 15% of the AAPL stock price?",
        expected_tools=["get_stock_price", "calculate"],
        expected_keywords=["AAPL", "%"],
    ),
    AgentTestCase(
        question="Tell me about electric vehicles",
        expected_tools=["web_search"],
        expected_keywords=["million", "2023"],
    ),
]

test_agent(TEST_SUITE)
```

> [!success] Test your agent with expected tool sequences, not just expected outputs
> Output text is easy to hallucinate correctly. If your agent answers the right question by bypassing its tools, it will fail on novel inputs. Test that the right tools were called in addition to checking the output.

---

[[03-planning-strategies]] | [[05-memory-strategies]]
