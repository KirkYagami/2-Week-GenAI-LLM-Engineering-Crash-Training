# The ReAct Loop

ReAct (Reasoning + Acting) is the pattern that gives agents a visible thought process before each action. Instead of calling a tool immediately, the model first writes a Thought — its internal reasoning about what to do — then specifies an Action. After executing the action, it sees the Observation and repeats. This trace is both the agent's scratchpad and your debugging surface.

## Learning objectives

- Implement the Thought → Action → Observation loop from scratch
- Parse and format ReAct traces for the model
- Know when to use ReAct vs pure tool-calling loops
- Read a ReAct trace to diagnose agent failures
- Handle the `Final Answer` termination condition

---

## The ReAct format

```
User: What's the population of the capital of France?

Thought: I need to find the capital of France first.
Action: search[capital of France]
Observation: The capital of France is Paris.

Thought: Now I need the population of Paris.
Action: search[population of Paris]
Observation: Paris has a population of approximately 2.16 million (city proper).

Thought: I have all the information needed.
Final Answer: The population of Paris, the capital of France, is approximately 2.16 million people.
```

The format provides a reasoning trace that's inspectable. If the agent goes wrong, you can see exactly which thought led to the wrong action.

---

## ReAct prompt setup

```python
import os
import json
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REACT_SYSTEM_PROMPT = """You are a reasoning agent. Solve problems step by step using this format:

Thought: [your reasoning about what to do next]
Action: tool_name[input]
Observation: [the result will be inserted here]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information.
Final Answer: [your complete answer to the original question]

Available tools:
{tool_descriptions}

Rules:
- Always start with Thought:
- Use exactly one Action: per step
- Wait for the Observation before your next Thought
- End with "Final Answer:" when done
"""

def format_tool_descriptions(tools: dict[str, callable]) -> str:
    descriptions = []
    for name, func in tools.items():
        doc = func.__doc__ or "No description"
        descriptions.append(f"- {name}: {doc.strip()}")
    return "\n".join(descriptions)
```

---

## Full ReAct implementation

```python
import os
import json
import re
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tool implementations
def search(query: str) -> str:
    """Search the web for information about a topic."""
    # Simulated search results
    knowledge = {
        "capital of france": "The capital of France is Paris.",
        "population of paris": "Paris has approximately 2.16 million people (city proper), 10.8M in metro area.",
        "eiffel tower height": "The Eiffel Tower is 330 meters (1,083 ft) tall including the antenna.",
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "llm meaning": "LLM stands for Large Language Model, a type of neural network trained on text.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if any(word in query_lower for word in key.split()):
            return value
    return f"No specific information found for: {query}"

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: arithmetic expression like '2 + 2' or '100 * 0.15'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

TOOLS = {"search": search, "calculator": calculator}

SYSTEM_PROMPT = f"""You are a reasoning agent. Solve problems step by step.

Format:
Thought: [reasoning]
Action: tool_name[input]
Observation: [result inserted]
... repeat ...
Final Answer: [complete answer]

Tools:
- search[query]: Search for information
- calculator[expression]: Evaluate math

Always start with Thought: and end with Final Answer:"""

def parse_action(text: str) -> tuple[str, str] | None:
    """Extract tool name and input from 'Action: tool_name[input]'"""
    match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None

def is_final(text: str) -> str | None:
    """Extract the final answer if present."""
    match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    return match.group(1).strip() if match else None

def react_agent(question: str, max_steps: int = 8) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stop=["Observation:"],  # Stop before writing its own observation
            temperature=0.0,
            max_tokens=300,
        )
        agent_output = response.choices[0].message.content
        print(f"\n[Step {step + 1}]\n{agent_output}")

        # Check if we've reached a final answer
        final = is_final(agent_output)
        if final:
            return final

        # Parse the action
        action = parse_action(agent_output)
        if not action:
            return f"Agent failed to produce a valid action. Last output:\n{agent_output}"

        tool_name, tool_input = action
        if tool_name not in TOOLS:
            observation = f"Error: Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
        else:
            observation = TOOLS[tool_name](tool_input)

        print(f"Observation: {observation}")

        # Add the agent's output + the observation back into the conversation
        messages.append({"role": "assistant", "content": agent_output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Max steps reached. Unable to complete the task."

# Test
print("=" * 60)
result = react_agent("What is 15% of the population of the capital of France?")
print(f"\nFinal Answer: {result}")
```

---

## Reading a ReAct trace

```python
# A good trace looks like:
GOOD_TRACE = """
Thought: I need to find the population first.
Action: search[population of Paris]
Observation: Paris has approximately 2.16 million people.

Thought: Now I calculate 15% of 2,160,000.
Action: calculator[2160000 * 0.15]
Observation: 2160000 * 0.15 = 324000

Thought: I have the answer.
Final Answer: 15% of Paris's population is approximately 324,000 people.
"""

# A bad trace shows these warning signs:
BAD_TRACE_SIGNS = [
    "Action: search[the answer to the question]  # Too vague",
    "Thought: I already know this... Final Answer: [hallucination]  # Skipped tools",
    "Action: unknown_tool[input]  # Hallucinated tool name",
    "Thought: ... Thought: ... Thought: ...  # Repeating without actions",
]
```

> [!tip] Use `stop=["Observation:"]` to prevent the model from writing its own observations
> Without the stop sequence, the model will write both the Action *and* the Observation itself, bypassing your tool execution entirely. Always stop before the Observation and inject the real tool result.

---

## ReAct vs pure tool-calling: when to use each

| Situation | Prefer |
|-----------|--------|
| Multi-step reasoning where steps depend on prior results | ReAct |
| Simple tool dispatch (weather + stock + exchange) | Tool-calling loop |
| Debugging agent behavior | ReAct (trace is inspectable) |
| Latency-sensitive production | Tool-calling (fewer tokens) |
| Tasks requiring visible reasoning for trust | ReAct |

---

[[01-what-are-agents]] | [[03-planning-strategies]]
