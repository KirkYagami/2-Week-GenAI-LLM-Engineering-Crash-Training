# Nodes and Edges

Nodes do computation. Edges control flow. Every LangGraph application is built from these two primitives — and understanding their rules determines whether your graph compiles and runs correctly.

## Learning objectives

- Write node functions with the correct signature
- Add nodes and edges to a `StateGraph`
- Use `START` and `END` sentinels
- Chain nodes into a sequential pipeline
- Add tool nodes with `ToolNode`

---

## Node rules

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    value: int
    history: list[str]

# Rule 1: Node receives the full state, returns a partial update (dict)
def double_node(state: State) -> dict:
    return {"value": state["value"] * 2}  # Only returns what changed

# Rule 2: Return value is MERGED into state, not replaced
# If you return {"value": 10}, the state becomes:
# {"value": 10, "history": [existing history]}  ← history is preserved

# Rule 3: Nodes can return any subset of state keys
def log_node(state: State) -> dict:
    return {"history": state["history"] + [f"value was {state['value']}"]}

# Rule 4: Async nodes are supported
async def async_node(state: State) -> dict:
    import asyncio
    await asyncio.sleep(0)
    return {"value": state["value"] + 1}
```

---

## Building a sequential pipeline

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

class WritingState(TypedDict):
    topic: str
    outline: str
    draft: str
    final: str

def outline_node(state: WritingState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Create a 3-bullet outline for an article."),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    return {"outline": response.content}

def draft_node(state: WritingState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Write a 2-paragraph article based on this outline."),
        HumanMessage(content=f"Topic: {state['topic']}\nOutline:\n{state['outline']}")
    ])
    return {"draft": response.content}

def polish_node(state: WritingState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Polish this draft: fix grammar, improve flow, keep it concise."),
        HumanMessage(content=state["draft"])
    ])
    return {"final": response.content}

# Build the graph
graph = StateGraph(WritingState)
graph.add_node("outline", outline_node)
graph.add_node("draft", draft_node)
graph.add_node("polish", polish_node)

# Connect nodes sequentially
graph.add_edge(START, "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "polish")
graph.add_edge("polish", END)

app = graph.compile()

result = app.invoke({"topic": "The benefits of local LLMs", "outline": "", "draft": "", "final": ""})
print("Outline:", result["outline"][:200])
print("\nFinal:", result["final"][:300])
```

---

## Edges at a glance

```python
# Direct edge (unconditional)
graph.add_edge("node_a", "node_b")

# Edge from START (entry point)
graph.add_edge(START, "first_node")

# Edge to END (exit)
graph.add_edge("last_node", END)

# Conditional edge (covered in 04-conditional-routing)
graph.add_conditional_edges(
    "router_node",
    router_function,           # Returns a node name
    {"option_a": "node_a", "option_b": "node_b"},
)
```

---

## Tool nodes with `ToolNode`

LangGraph provides a `ToolNode` that handles the tool-calling pattern (execute all tool calls in the assistant message, return results) automatically:

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
import operator

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Define tools as LangChain tool objects
@tool
def search(query: str) -> str:
    """Search for information on a topic."""
    data = {
        "python": "Python is a high-level, interpreted programming language.",
        "rust": "Rust is a systems programming language focused on safety and performance.",
    }
    for key, val in data.items():
        if key in query.lower():
            return val
    return f"No information found for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

tools = [search, calculate]
llm_with_tools = llm.bind_tools(tools)

# State that accumulates messages
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # operator.add appends lists

def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")  # After tools, go back to agent

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="Search for Python and calculate 2^10")]
})
print(result["messages"][-1].content)
```

> [!tip] `operator.add` as a reducer appends lists rather than replacing them
> When state has `Annotated[list[BaseMessage], operator.add]`, returning `{"messages": [new_message]}` appends the new message to the existing list. Without the annotation, returning `{"messages": [...]}` would replace the entire list.

---

[[01-langgraph-overview]] | [[03-stateful-graphs]]
