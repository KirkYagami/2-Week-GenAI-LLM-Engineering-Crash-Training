# Planning Strategies

ReAct is a reactive planner — it decides one step at a time. For longer, structured tasks, you get better results by generating a plan upfront and then executing it. This note covers three planning strategies and when each outperforms the others.

## Learning objectives

- Implement sequential plan-and-execute
- Parallelize independent plan steps with asyncio
- Apply the plan → execute → reflect loop for self-improving agents
- Know the failure modes of each planning approach

---

## Sequential plan-and-execute

Generate the full plan first, then execute each step in order. Best for tasks with a known structure.

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_plan(task: str, available_tools: list[str]) -> list[dict]:
    """Generate a step-by-step execution plan."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Create a step-by-step plan to complete this task.
Task: {task}
Available tools: {available_tools}

Return JSON: {{"steps": [{{"step_id": 1, "description": "...", "tool": "tool_name", "input": "..."}}]}}
Keep it to 3-5 steps maximum."""
        }],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content).get("steps", [])

def execute_step(step: dict, tool_map: dict, context: dict) -> str:
    """Execute a single plan step, substituting context variables."""
    tool_name = step["tool"]
    tool_input = step["input"]

    # Simple variable substitution from previous step results
    for key, value in context.items():
        tool_input = tool_input.replace(f"{{{key}}}", str(value))

    if tool_name not in tool_map:
        return f"Error: unknown tool {tool_name}"
    return str(tool_map[tool_name](tool_input))

def plan_and_execute(task: str, tool_map: dict) -> str:
    available = list(tool_map.keys())
    plan = generate_plan(task, available)

    print(f"Plan ({len(plan)} steps):")
    for step in plan:
        print(f"  {step['step_id']}. [{step['tool']}] {step['description']}")

    context = {}
    results = []
    for step in plan:
        result = execute_step(step, tool_map, context)
        context[f"step_{step['step_id']}_result"] = result
        results.append({"step": step["description"], "result": result})
        print(f"\nStep {step['step_id']} result: {result[:100]}")

    # Synthesize final answer
    synthesis_prompt = f"""Task: {task}
Execution results:
{json.dumps(results, indent=2)}

Provide a concise final answer based on these results."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.0, max_tokens=300,
    )
    return resp.choices[0].message.content

# Test tools
def web_search(query: str) -> str:
    data = {
        "nvidia revenue 2024": "NVIDIA reported $60.9 billion in revenue for fiscal year 2024.",
        "amd revenue 2024": "AMD reported $22.7 billion in revenue for fiscal year 2024.",
    }
    return data.get(query.lower(), f"No data for: {query}")

def calculator(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {"web_search": web_search, "calculator": calculator}
result = plan_and_execute("Compare NVIDIA and AMD revenue in 2024 and calculate the ratio", TOOLS)
print(f"\nFinal Answer:\n{result}")
```

---

## Parallel plan execution

When plan steps are independent, execute them concurrently:

```python
import asyncio
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def execute_step_async(step: dict, tool_map: dict) -> dict:
    """Execute one plan step asynchronously."""
    await asyncio.sleep(0)  # Yield control
    tool_name = step.get("tool", "")
    tool_input = step.get("input", "")
    if tool_name in tool_map:
        result = tool_map[tool_name](tool_input)
    else:
        result = f"Unknown tool: {tool_name}"
    return {"step_id": step["step_id"], "description": step["description"], "result": result}

async def parallel_execute(task: str, tool_map: dict) -> str:
    plan = generate_plan(task, list(tool_map.keys()))

    # Identify independent steps (no cross-dependencies in input)
    # Simple heuristic: steps that don't reference prior step results
    independent = [s for s in plan if "{step_" not in s.get("input", "")]
    dependent = [s for s in plan if "{step_" in s.get("input", "")]

    print(f"Parallel: {len(independent)} steps | Sequential: {len(dependent)} steps")

    # Execute independent steps in parallel
    parallel_results = await asyncio.gather(
        *[execute_step_async(s, tool_map) for s in independent]
    )

    # Build context from parallel results
    context = {f"step_{r['step_id']}_result": r["result"] for r in parallel_results}

    # Execute dependent steps sequentially
    sequential_results = []
    for step in dependent:
        result = execute_step(step, tool_map, context)
        context[f"step_{step['step_id']}_result"] = result
        sequential_results.append({"step": step["description"], "result": result})

    all_results = list(parallel_results) + sequential_results
    return f"Completed {len(all_results)} steps"

asyncio.run(parallel_execute("Research both NVIDIA and AMD revenue then compare them", TOOLS))
```

---

## Plan → Execute → Reflect

Add a reflection step to catch errors and refine the plan:

```python
def plan_execute_reflect(task: str, tool_map: dict, max_iterations: int = 3) -> str:
    plan = generate_plan(task, list(tool_map.keys()))
    context = {}

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")

        # Execute all steps
        results = []
        failed_steps = []
        for step in plan:
            result = execute_step(step, tool_map, context)
            context[f"step_{step['step_id']}_result"] = result
            results.append({"step": step["description"], "result": result})
            if "Error" in result or "not found" in result.lower():
                failed_steps.append(step)

        if not failed_steps:
            print("All steps succeeded!")
            break

        # Reflect and revise the plan
        print(f"Failed steps: {[s['description'] for s in failed_steps]}")
        reflection_prompt = f"""Task: {task}
Plan execution results:
{json.dumps(results, indent=2)}

Some steps failed. Revise the plan to fix the failures.
Return JSON: {{"steps": [{{"step_id": ..., "description": "...", "tool": "...", "input": "..."}}]}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": reflection_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        plan = json.loads(resp.choices[0].message.content).get("steps", plan)
        print(f"Plan revised with {len(plan)} steps")

    # Final synthesis
    final_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Task: {task}\nResults: {json.dumps(results, indent=2)}\nProvide a final answer."}],
        temperature=0.0, max_tokens=300,
    )
    return final_resp.choices[0].message.content
```

---

## Planning strategy comparison

| Strategy | Steps known upfront | Handles surprises | Parallelism | Best for |
|----------|---------------------|-------------------|-------------|----------|
| ReAct (reactive) | No | Yes | No | Unknown-length tasks |
| Plan-and-execute | Yes | No | Partial | Structured tasks |
| Parallel plan | Yes | No | Yes | Independent subtasks |
| Plan-execute-reflect | Yes | Yes | Partial | Long tasks needing reliability |

> [!tip] Start with plan-and-execute, add reflection only if failure rate is high
> Planning adds latency (one extra LLM call). Reflection adds more. Start with the simplest approach that achieves acceptable reliability. Add complexity only when measured failure rate justifies it.

---

[[02-react-loop]] | [[04-tool-calling-agents]]
