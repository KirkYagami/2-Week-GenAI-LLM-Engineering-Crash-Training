# LangGraph Overview

A simple tool-calling loop works for most agents. But when you need branching logic, parallel execution, human-in-the-loop checkpoints, or persistent state across sessions, you're describing a graph — not a loop. LangGraph models agent behavior as a directed graph where nodes are computation units and edges are control flow.

## Learning objectives

- Understand why graphs solve problems that loops don't
- Know the three core primitives: nodes, edges, state
- Set up a minimal `StateGraph` and compile it
- Understand how LangGraph differs from LangChain LCEL

---

## The problem with plain loops

```
Plain agent loop:
  user → [model → tools → model → tools → ... → model] → answer

Works for:
  - Single goal, sequential reasoning
  - Unknown number of steps
  - Simple tool dispatch

Fails for:
  - Branching based on output quality ("retry if answer is too vague")
  - Parallel sub-tasks ("search topic A and topic B simultaneously")
  - Human approval gates ("pause here and wait for human review")
  - Persistent state across sessions ("resume where we left off")
  - Multiple specialized agents ("researcher → writer → critic → writer")
```

LangGraph represents each of these as graph topology, making complex control flow explicit and inspectable.

---

## The three primitives

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. STATE: the shared data structure that flows through the graph
class MyState(TypedDict):
    input: str
    result: str
    steps: int

# 2. NODES: functions that take state and return updated state
def process_node(state: MyState) -> dict:
    return {"result": f"Processed: {state['input']}", "steps": state["steps"] + 1}

# 3. EDGES: connections between nodes (direct or conditional)
graph = StateGraph(MyState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"input": "hello", "steps": 0})
print(result)
# {'input': 'hello', 'result': 'Processed: hello', 'steps': 1}
```

---

## LangGraph vs LCEL

| Aspect | LCEL | LangGraph |
|--------|------|-----------|
| Control flow | Sequential pipeline | Arbitrary directed graph |
| Branching | Not supported | Conditional edges |
| Loops | Not supported | Cycles allowed |
| State | Passed as function args | Typed shared state object |
| Human-in-loop | Not supported | Interrupt/resume |
| Persistence | Not supported | Checkpointing |
| Best for | Linear chains | Stateful, complex agents |

LCEL is for simple pipelines. LangGraph is for agents that need to loop, branch, or involve multiple models.

---

## Minimal complete example

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

class ChatState(TypedDict):
    messages: list[dict]
    turn_count: int

def chat_node(state: ChatState) -> dict:
    from langchain_core.messages import HumanMessage, AIMessage

    # Convert our dict messages to LangChain format
    lc_messages = []
    for m in state["messages"]:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    response = llm.invoke(lc_messages)
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
        "turn_count": state["turn_count"] + 1,
    }

# Build the graph
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)
app = graph.compile()

# Run
initial_state = {
    "messages": [{"role": "user", "content": "What is LangGraph?"}],
    "turn_count": 0,
}
result = app.invoke(initial_state)
print(f"Turn: {result['turn_count']}")
print(f"Answer: {result['messages'][-1]['content']}")
```

> [!info] LangGraph uses LangChain's message types internally
> LangGraph works best with `HumanMessage`, `AIMessage`, `SystemMessage` from `langchain_core.messages`. You can use plain dicts in state but convert to LangChain types when calling LLMs.

---

## The compilation step

```python
# Compiling creates an immutable Runnable
app = graph.compile()

# Supports all Runnable methods
result = app.invoke(state)                    # Synchronous
result = await app.ainvoke(state)             # Async
for chunk in app.stream(state):               # Stream state updates
    print(chunk)
```

The compiled graph validates: no orphan nodes, edges reference existing nodes, no structural contradictions. Errors surface at compile time, not runtime.

---

[[00-agenda]] | [[02-nodes-and-edges]]
