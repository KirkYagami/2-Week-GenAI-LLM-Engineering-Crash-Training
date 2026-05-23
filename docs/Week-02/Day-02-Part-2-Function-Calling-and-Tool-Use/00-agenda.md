# Agenda — Function Calling and Tool Use

**Session length:** 3 hours | **Difficulty:** Intermediate | **Prerequisites:** [[Week-02/Day-01-Part-1-LangChain-Fundamentals/04-lcel|LCEL]], OpenAI and Anthropic API basics

---

## What you will build today

A structured data extraction pipeline that uses tool use / function calling to reliably extract entities from unstructured text — no regex, no prompt parsing, just JSON schemas.

---

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00–0:20 | What function calling is and why it matters | [[01-function-calling-overview]] |
| 0:20–0:55 | OpenAI Tools API: defining tools, parsing calls, executing | [[02-openai-tools]] |
| 0:55–1:25 | Anthropic tool_use: same pattern, different API shape | [[03-anthropic-tool-use]] |
| 1:25–1:50 | Structured extraction with Pydantic and `with_structured_output` | [[04-structured-extraction]] |
| 1:50–2:10 | Parallel tool calls: multiple tools in one round | [[05-parallel-tool-calls]] |
| 2:10–2:45 | Practice exercises | [[06-practice-exercises]] |
| 2:45–3:00 | Interview questions review | [[07-interview-questions]] |

---

## Setup

```bash
pip install openai anthropic pydantic langchain-openai
```

---

[[Week-02/Day-02-Part-1-Fine-Tuning/07-interview-questions|← Fine-Tuning]] | [[01-function-calling-overview|Overview →]]
