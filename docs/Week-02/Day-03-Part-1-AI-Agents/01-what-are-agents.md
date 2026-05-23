# What Are AI Agents

A chain runs a fixed sequence of steps. An agent decides its own steps. That single distinction — the model choosing what to do next rather than following a predetermined path — is what makes agents powerful and what makes them hard to build reliably.

## Learning objectives

- Define what makes a system an "agent" vs a chain or pipeline
- Identify the three components every agent needs: model, tools, loop
- Know when agents are appropriate and when they're overkill
- Understand the core failure modes: loops, tool misuse, context loss

---

## Chains vs agents

```
Chain (fixed):
  User query → Retrieve → Compress → Generate → Return
  Every step is predetermined. No branching. No tool selection.

Agent (dynamic):
  User query → Model decides: "I need to search first"
              → Search → Model sees results → "I need a calculator now"
              → Calculate → Model sees output → "I have enough, I'll answer"
              → Return
  Steps are chosen at runtime. Number of steps is unknown in advance.
```

---

## The three components of every agent

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Agent:
    # 1. MODEL — the decision-maker
    model: str = "gpt-4o-mini"

    # 2. TOOLS — actions the agent can take
    tools: list[dict] = field(default_factory=list)
    tool_implementations: dict[str, Callable] = field(default_factory=dict)

    # 3. LOOP — the mechanism that keeps the agent running
    max_steps: int = 10  # Hard limit to prevent infinite loops

    # Optionally: memory, planning, critics, etc.
```

The loop is what distinguishes an agent from a single tool call. The model's output feeds back as input until it decides it's done.

---

## A taxonomy of agent designs

```python
AGENT_TYPES = {
    "ReAct": {
        "pattern": "Thought → Action → Observation → repeat",
        "strength": "Transparent reasoning; easy to debug",
        "weakness": "Verbose; slow for simple tasks",
        "use_case": "Research, Q&A requiring multiple lookups",
    },
    "Tool-calling loop": {
        "pattern": "Call tools → aggregate results → answer",
        "strength": "Fast; uses native API tool support",
        "weakness": "No explicit reasoning trace",
        "use_case": "Data retrieval, API orchestration",
    },
    "Plan-and-execute": {
        "pattern": "Generate full plan → execute each step",
        "strength": "Efficient; parallel execution possible",
        "weakness": "Plan can go stale; brittle to surprises",
        "use_case": "Long tasks with known structure",
    },
    "Multi-agent": {
        "pattern": "Orchestrator delegates to specialized agents",
        "strength": "Parallelism; specialization",
        "weakness": "Complex to orchestrate and debug",
        "use_case": "Research + writing + review pipelines",
    },
}
```

---

## When to use agents

```python
USE_AGENTS_WHEN = [
    "The number of steps needed isn't known in advance",
    "The output of one step determines what the next step should be",
    "Multiple tools must be selected and ordered dynamically",
    "The task involves iterative refinement (write → critique → rewrite)",
]

DON'T_USE_AGENTS_WHEN = [
    "The pipeline is fixed and always does the same steps",
    "Latency budget is tight (agents add multiple round trips)",
    "The task is simple enough for a single LLM call",
    "Reliability is critical and you can't tolerate agent hallucination",
]
```

> [!warning] Agents are not always the answer
> The appeal of agents is flexibility. The cost is unpredictability. An agent that works 90% of the time is a liability in a production system where users depend on consistency. Reserve agents for tasks that genuinely require dynamic decision-making; use chains for everything that doesn't.

---

## Core failure modes

| Failure | Cause | Fix |
|---------|-------|-----|
| Infinite loop | Agent keeps calling tools without reaching stop | Add `max_steps` counter |
| Tool hallucination | Agent invents function names | Validate tool names before dispatch |
| Context overflow | Too many tool results fill the context window | Summarize or truncate tool outputs |
| Goal drift | Agent wanders off-task after multiple steps | Remind agent of original goal each step |
| Cascading errors | Early wrong tool call poisons later steps | Add verification after critical steps |

---

## Minimal agent skeleton

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_agent(
    user_message: str,
    tools: list[dict],
    function_map: dict,
    system_prompt: str = "You are a helpful assistant.",
    max_steps: int = 10,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                if tc.function.name not in function_map:
                    result = {"error": f"Unknown tool: {tc.function.name}"}
                else:
                    args = json.loads(tc.function.arguments)
                    result = function_map[tc.function.name](**args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

    return "Max steps reached without a final answer."
```

> [!success] Add max_steps to every agent you build
> No exception. A missing `max_steps` guard turns a misbehaving agent into an API cost spiral. Set it based on your task: 5 steps for simple queries, 10–15 for research tasks, 20+ only with a human-in-the-loop checkpoint.

---

[[00-agenda]] | [[02-react-loop]]
