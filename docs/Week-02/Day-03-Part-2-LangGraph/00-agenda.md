# Agenda — LangGraph

**Session length:** 3 hours | **Difficulty:** Advanced | **Prerequisites:** [[Week-02/Day-03-Part-1-AI-Agents/04-tool-calling-agents|Tool-Calling Agents]], Python dataclasses, TypedDict

---

## What you will build today

A stateful multi-agent research pipeline: a planner node that decomposes a question, parallel researcher nodes that gather data, and a writer node that synthesizes the final report — with conditional routing based on quality checks.

---

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00–0:20 | What LangGraph is and why it exists | [[01-langgraph-overview]] |
| 0:20–0:50 | Nodes, edges, and the StateGraph API | [[02-nodes-and-edges]] |
| 0:50–1:20 | Stateful graphs: TypedDict state, reducers | [[03-stateful-graphs]] |
| 1:20–1:50 | Conditional routing: edges based on state | [[04-conditional-routing]] |
| 1:50–2:15 | Multi-agent orchestration | [[05-multi-agent-orchestration]] |
| 2:15–2:45 | Practice exercises | [[06-practice-exercises]] |
| 2:45–3:00 | Interview questions review | [[07-interview-questions]] |

---

## Setup

```bash
pip install langgraph langchain-openai langchain-core
```

---

[[Week-02/Day-03-Part-1-AI-Agents/07-interview-questions|← AI Agents]] | [[01-langgraph-overview|LangGraph Overview →]]
