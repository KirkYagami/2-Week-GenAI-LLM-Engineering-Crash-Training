# Agenda — AI Agents

**Session length:** 3 hours | **Difficulty:** Advanced | **Prerequisites:** [[Week-02/Day-02-Part-2-Function-Calling-and-Tool-Use/02-openai-tools|OpenAI Tools]], Python dataclasses, basic async

---

## What you will build today

A ReAct agent that loops through Thought → Action → Observation cycles to answer multi-step research questions, with three different memory strategies.

---

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00–0:20 | What agents are, when they're needed vs chains | [[01-what-are-agents]] |
| 0:20–0:50 | ReAct loop: Thought, Action, Observation | [[02-react-loop]] |
| 0:50–1:15 | Planning strategies: sequential, parallel, tree-of-thought | [[03-planning-strategies]] |
| 1:15–1:45 | Tool-calling agents: building a real agent from scratch | [[04-tool-calling-agents]] |
| 1:45–2:05 | Memory strategies: in-context, summarized, vector | [[05-memory-strategies]] |
| 2:05–2:45 | Practice exercises | [[06-practice-exercises]] |
| 2:45–3:00 | Interview questions review | [[07-interview-questions]] |

---

## Setup

```bash
pip install openai anthropic langgraph langchain-openai
```

---

[[Week-02/Day-02-Part-2-Function-Calling-and-Tool-Use/07-interview-questions|← Day 2 Part 2]] | [[01-what-are-agents|What Are Agents →]]
