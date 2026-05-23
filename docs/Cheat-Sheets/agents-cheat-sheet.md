# AI Agents Cheat Sheet

Quick reference for agent patterns: ReAct, tool registration, and memory.

---

## ReAct loop (manual)

```python
REACT_SYSTEM = """You solve tasks by thinking and using tools. Follow this format exactly:

Thought: <your reasoning about what to do next>
Action: <tool_name>({"param": "value"})
Observation: <result of the action — provided by the system>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: <your final answer>"""

messages = [
    {"role": "system", "content": REACT_SYSTEM},
    {"role": "user", "content": task},
]

for _ in range(max_steps):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stop=["Observation:"],  # Stop before generating fake observations
        temperature=0.0,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    if "Final Answer:" in reply:
        break

    # Execute the action
    action, args = parse_action(reply)
    result = tools[action](**args)
    messages.append({"role": "user", "content": f"Observation: {result}"})
```

## Tool registration pattern

```python
tools = {}

def tool(name: str, description: str, **param_descriptions):
    def decorator(func):
        tools[name] = func
        func.__tool_def__ = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param: {"type": "string", "description": desc}
                        for param, desc in param_descriptions.items()
                    },
                    "required": list(param_descriptions.keys()),
                },
            },
        }
        return func
    return decorator

@tool("search_web", "Search the web for current information", query="The search query")
def search_web(query: str) -> str:
    # ... implementation
    return f"Results for: {query}"

tool_defs = [f.__tool_def__ for f in tools.values()]
```

## Memory strategies

```python
# In-context (full history — simplest)
memory = []
memory.append({"role": "user", "content": user_msg})
memory.append({"role": "assistant", "content": reply})

# Sliding window (last N messages)
MAX_MESSAGES = 20
if len(memory) > MAX_MESSAGES:
    memory = memory[-MAX_MESSAGES:]

# Summary buffer (compress old messages)
async def summarize_and_compress(memory: list[dict]) -> list[dict]:
    old = memory[:-10]
    recent = memory[-10:]
    summary = await llm.ainvoke([
        {"role": "user", "content": f"Summarize this conversation:\n{format(old)}"}
    ])
    return [
        {"role": "system", "content": f"Previous conversation summary: {summary}"},
        *recent
    ]

# Vector memory (semantic retrieval)
import numpy as np
def retrieve_relevant_memories(query: str, memories: list[dict], top_k: int = 3) -> list[dict]:
    q_emb = embed(query)
    scored = [(float(q_emb @ m["emb"]), m) for m in memories]
    return [m for _, m in sorted(scored, reverse=True)[:top_k]]
```

## Safety guards

```python
# Max steps guard
for step in range(max_steps := 10):
    ...
    if step == max_steps - 1:
        return "Max steps reached. Could not complete task."

# Token budget guard
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
if sum(len(enc.encode(m["content"])) for m in messages) > 100_000:
    messages = compress_memory(messages)
```

## Agent vs. chain decision

Use an **agent** when: the number of steps needed is unknown, tools need to be selected dynamically, or the output of one tool determines which tool to call next.

Use a **chain** when: the steps are fixed and known in advance, no dynamic tool selection is needed.
