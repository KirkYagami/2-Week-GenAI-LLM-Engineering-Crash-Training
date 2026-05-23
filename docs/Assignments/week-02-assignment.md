# Week 02 Assignment

The Week 2 assignment is your capstone project. Complete it using the guidelines in the [[Day-05-Part-1-Capstone-Project]] and submit via GitHub.

---

## Option A — Full capstone (recommended)

Build one of the three capstone options from [Day 5 Part 1](../Week-02/Day-05-Part-1-Capstone-Project/01-project-brief.md) to production quality:

- Option A: Customer Support Bot with RAG
- Option B: Research Agent with LangGraph
- Option C: Document Intelligence Pipeline

### Requirements

**Must include (required):**
- [ ] At least 4 course components integrated
- [ ] Working FastAPI endpoint (not just a script)
- [ ] Async patterns used correctly
- [ ] Error handling for API failures and invalid input
- [ ] At least one quantitative evaluation metric with results
- [ ] README with setup instructions and eval results

**Stretch goals (bonus):**
- [ ] Streaming response via SSE
- [ ] LangSmith tracing enabled
- [ ] Deployed on Fly.io, Modal, or Railway (provide the URL)
- [ ] RAGAS evaluation (RAG projects) or LLM-as-judge (agent projects)
- [ ] Cost tracking (tokens + estimated cost per request)

---

## Option B — Extended project from Week 1

Extend your Week 1 RAG assignment to production quality:
- Add a FastAPI endpoint with proper validation
- Add streaming
- Add evaluation (at least Recall@3 + RAGAS faithfulness)
- Add a cache layer

Must still meet the Week 2 required component count (4+).

---

## Submission checklist

Before submitting, verify every item:

- [ ] `README.md` has: what it does, how to run it (2 commands), eval results
- [ ] `.gitignore` includes `.env` and `chroma_db/`
- [ ] No hardcoded API keys anywhere in commit history
- [ ] `.env.example` provided with blank values
- [ ] `requirements.txt` with pinned versions
- [ ] At least one eval script that runs and prints numbers

Submit: GitHub repository URL in the course platform.

---

## Grading rubric

| Criterion | Points | Pass criteria |
|-----------|--------|--------------|
| Integration depth | 15 | 4+ components, wired correctly |
| Code quality | 10 | Async, error handling, no hardcoded keys |
| Working demo | 20 | Endpoint runs and handles happy path + one edge case |
| Evaluation | 25 | At least one metric with ≥10 test cases, results in README |
| Architecture clarity | 15 | README or DESIGN.md explains data flow and component choices |
| Stretch goals | 15 | Each item from the stretch list = 3 points |

Total: 85 base + 15 stretch = 100 points. Pass threshold: 65/100.

---

## Submission deadline

Due before the final session (Day 5 Part 2 — Mock Interview and Portfolio Review). You will demo your project live in that session.

> [!success] The goal of this assignment
> Build something you'd be proud to show in an interview. A working, evaluated, documented project that you built from scratch is more valuable on your resume than any certificate.
