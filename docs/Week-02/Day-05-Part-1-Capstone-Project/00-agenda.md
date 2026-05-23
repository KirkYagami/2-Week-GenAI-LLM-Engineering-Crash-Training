# Agenda — Day 5 Part 1: End-to-End Capstone Project

The capstone is where two weeks of concepts become a single working system. You will design, build, evaluate, and present an LLM application that integrates at least four of the major components covered in this course.

## Session goals

By the end of this session you will have:

- A deployed, working LLM application connected to real APIs
- A documented architecture with rationale for every design decision
- A quantitative evaluation suite with at least two metrics
- A 5-minute demo ready to present to the group

---

## Schedule

| Time | Activity |
|------|----------|
| 0:00–0:20 | Project brief + scope selection |
| 0:20–0:50 | Architecture design session (whiteboard / doc) |
| 0:50–2:30 | Implementation sprint |
| 2:30–3:00 | Evaluation + metrics run |
| 3:00–3:30 | Demo prep + presentation |
| 3:30–4:00 | Group demos + feedback |

---

## Required components (choose at least 4)

- [ ] RAG pipeline (chunking + embedding + retrieval)
- [ ] LLM API calls with proper error handling and retries
- [ ] Streaming response delivery
- [ ] Function calling / tool use
- [ ] LangGraph agent with conditional routing
- [ ] Async FastAPI service
- [ ] Caching layer (exact-match or semantic)
- [ ] Evaluation suite (RAGAS, custom metrics, or LangSmith evals)
- [ ] Cost tracking and budget guard
- [ ] Deployment (Modal, Fly.io, or Docker + uvicorn)

---

## Deliverables

1. **Working code** — runnable from a `README.md` with two commands
2. **Architecture diagram** — hand-drawn, Mermaid, or any tool
3. **Eval results** — numbers, not vibes
4. **5-minute demo** — live or recorded

---

[[01-project-brief]] | [[02-architecture-design]]
