# LangGraph Cheat Sheet

Quick reference for LangGraph StateGraph API.

---

## Minimal graph

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class State(TypedDict):
    message: str
    count: int

def node_a(state: State) -> dict:
    return {"message": "Hello from A", "count": state["count"] + 1}

def node_b(state: State) -> dict:
    return {"message": state["message"] + " and B"}

graph = StateGraph(State)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.set_entry_point("a")
graph.add_edge("a", "b")
graph.add_edge("b", END)
app = graph.compile()

result = app.invoke({"message": "", "count": 0})
```

## Reducer (accumulating list)

```python
import operator
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list[str], operator.add]  # Each node appends, not overwrites

def add_message(state: State) -> dict:
    return {"messages": ["new message"]}  # Appended to existing list
```

## Conditional routing

```python
def router(state: State) -> str:
    if state["quality_score"] >= 0.8:
        return "approve"
    elif state["attempts"] >= 3:
        return "force_approve"
    else:
        return "revise"

graph.add_conditional_edges(
    "critic",
    router,
    {"approve": "finalize", "force_approve": "finalize", "revise": "writer"},
)
```

## Async nodes

```python
async def async_node(state: State) -> dict:
    result = await some_async_call(state["input"])
    return {"output": result}

# Invoke async:
result = await app.ainvoke(initial_state)
```

## MemorySaver checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Thread ID ties state to a session
config = {"configurable": {"thread_id": "session-123"}}
result = await app.ainvoke(state, config=config)

# Read state
current = app.get_state(config)
print(current.values)

# Update state manually
app.update_state(config, {"message": "manually edited"})

# Resume (after interrupt)
result = await app.ainvoke(None, config=config)
```

## Human-in-the-loop

```python
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["critic"],   # Pause before critic runs
    # interrupt_after=["writer"],  # Pause after writer runs
)
```

## Streaming events

```python
async for event in app.astream_events(state, version="v1"):
    if event["event"] == "on_chain_start":
        print(f"Starting: {event['name']}")
    elif event["event"] == "on_chain_end":
        print(f"Done: {event['name']}")
```

## Common patterns

```python
# Tool node (auto-executes tool calls from LLM)
from langgraph.prebuilt import ToolNode, tools_condition

tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)

graph.add_node("tools", tool_node)
graph.add_conditional_edges("agent", tools_condition)  # Routes to tools or END

# Sub-graph
subgraph_app = subgraph.compile()
graph.add_node("sub", subgraph_app)  # Entire compiled graph as a node
```

## Graph visualization

```python
from IPython.display import Image
Image(app.get_graph().draw_mermaid_png())
```
