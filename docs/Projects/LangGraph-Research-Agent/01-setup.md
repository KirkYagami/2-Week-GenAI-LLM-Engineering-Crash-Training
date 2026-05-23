# Setup and Environment — LangGraph Research Agent

## Project structure

```
research-agent/
├── agent.py            ← LangGraph graph definition
├── nodes.py            ← Node functions (planner, researcher, writer, critic)
├── state.py            ← State TypedDict definition
├── app.py              ← FastAPI endpoint
├── eval.py             ← Agent evaluation script
├── requirements.txt
├── .env.example
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```
# requirements.txt
langchain==0.3.7
langchain-openai==0.2.3
langchain-core==0.3.15
langgraph==0.2.28
langsmith==0.1.147
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
python-dotenv==1.0.1
httpx==0.27.2
```

```bash
# .env.example
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls-...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=research-agent
```

## Verify LangGraph installation

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class TestState(TypedDict):
    message: str

graph = StateGraph(TestState)
graph.add_node("hello", lambda s: {"message": "Hello from LangGraph!"})
graph.set_entry_point("hello")
graph.add_edge("hello", END)
app = graph.compile()
result = app.invoke({"message": ""})
print(result["message"])  # Hello from LangGraph!
```

---

[[README]] | [[02-implementation]]
