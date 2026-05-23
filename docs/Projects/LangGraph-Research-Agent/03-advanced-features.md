# Advanced Features — LangGraph Research Agent

## MemorySaver: persist state across sessions

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
research_app = graph.compile(checkpointer=memory)

# Each run with the same thread_id continues the same conversation
config = {"configurable": {"thread_id": "user-123"}}

result1 = await research_app.ainvoke({"question": "What is RAG?"}, config=config)
result2 = await research_app.ainvoke({"question": "How does it compare to fine-tuning?"}, config=config)

# Inspect state at any point
state = research_app.get_state(config)
print(state.values["final_report"])
```

## LangSmith tracing

Set the environment variables and LangSmith automatically captures every node execution:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=research-agent
```

Every `research_app.ainvoke()` call creates a traced run in LangSmith showing:
- Each node's input state and output state
- Token usage per node
- Wall clock time per node
- The conditional routing decision

## Adding a web search tool

Replace the LLM-only researcher node with real web search using Tavily:

```python
# pip install tavily-python
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

async def researcher_with_search(state: ResearchState) -> dict:
    async def research_one(sub_q: str) -> str:
        # Real web search
        try:
            search_result = tavily.search(query=sub_q, max_results=3)
            sources = "\n\n".join(
                f"[{r['url']}]\n{r['content']}"
                for r in search_result.get("results", [])
            )
        except Exception:
            sources = "Search unavailable."

        # Synthesize with LLM
        resp = await _llm_fast.ainvoke([
            SystemMessage(content="Synthesize these search results into a comprehensive answer. Cite URLs."),
            HumanMessage(content=f"Question: {sub_q}\n\nSources:\n{sources}"),
        ])
        return f"### {sub_q}\n\n{resp.content}"

    findings = await asyncio.gather(*[research_one(q) for q in state["sub_questions"]])
    return {"findings": list(findings)}
```

## Streaming node progress

Stream each node completion event to the client:

```python
import json
from fastapi.responses import StreamingResponse

@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    async def event_stream():
        async for event in research_app.astream_events(
            {"question": request.question, "sub_questions": [], "findings": [],
             "draft": "", "quality_score": 0.0, "feedback": "", "attempts": 0, "final_report": ""},
            version="v1",
        ):
            kind = event["event"]
            if kind == "on_chain_start" and "name" in event:
                yield f"data: {json.dumps({'event': 'node_start', 'node': event['name']})}\n\n"
            elif kind == "on_chain_end" and event["name"] == "finalize":
                output = event["data"]["output"]
                yield f"data: {json.dumps({'event': 'done', 'report': output.get('final_report', '')})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

[[02-implementation]] | [[04-evaluation]]
