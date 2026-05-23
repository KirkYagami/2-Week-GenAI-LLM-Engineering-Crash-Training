# Submission Checklist

Use this checklist to verify your capstone project before submission. Every item marked required must be complete. Items marked stretch are optional but improve your portfolio.

---

## Code quality

- [ ] **Required** — No hardcoded API keys. All secrets loaded via `os.getenv()`.
- [ ] **Required** — `requirements.txt` or `pyproject.toml` with pinned versions.
- [ ] **Required** — `README.md` with: what the project does, setup instructions (2–3 commands), example usage.
- [ ] **Required** — At least one working FastAPI endpoint with Pydantic request/response models.
- [ ] **Required** — Error handling for API failures (`try/except`, HTTP error codes, meaningful messages).
- [ ] **Stretch** — Dockerfile or deployment config (Modal, Fly.io) included.
- [ ] **Stretch** — `.env.example` showing required environment variables (values blanked out).

---

## Architecture and integration

- [ ] **Required** — At least 4 course components integrated (see [[00-agenda]] component list).
- [ ] **Required** — Architecture diagram included (`DESIGN.md` or `README.md`).
- [ ] **Required** — Async patterns used correctly (`AsyncOpenAI`, `await`, not blocking the event loop).
- [ ] **Stretch** — LangSmith tracing enabled and at least one trace screenshot in the README.
- [ ] **Stretch** — Cost tracking: total tokens and estimated cost logged per request.

---

## Evaluation

- [ ] **Required** — At least one quantitative metric defined and measured.
- [ ] **Required** — Eval script that runs against the live API and prints results.
- [ ] **Required** — At least 10 test inputs (20+ for a credible eval).
- [ ] **Stretch** — RAGAS evaluation (faithfulness + answer relevancy) for RAG projects.
- [ ] **Stretch** — Comparison of two approaches (e.g., chunking strategy A vs B, or with/without reranker).

---

## Presentation

- [ ] **Required** — 5-minute live demo or recording covering: problem, architecture, demo, eval results.
- [ ] **Required** — Can explain every design decision made.
- [ ] **Required** — Can identify the biggest current limitation and one concrete improvement.
- [ ] **Stretch** — Latency breakdown: time to first token, total time, tokens/second.

---

## Repository structure

Your submission repository should look like this:

```
my-capstone/
├── README.md              ← Setup, usage, results summary
├── DESIGN.md              ← Architecture decisions and data flow
├── requirements.txt       ← Pinned dependencies
├── .env.example           ← Required env vars (values blanked)
├── app.py                 ← Main FastAPI application
├── ingest.py              ← Data ingestion script (if RAG)
├── eval.py                ← Evaluation script
└── docs/                  ← Source documents for RAG (if applicable)
```

---

## Self-assessment rubric

Rate yourself honestly. This is for your own learning, not grading.

| Area | 1 — Not done | 2 — Partial | 3 — Complete | 4 — Polished |
|------|-------------|-------------|--------------|--------------|
| Integration depth | 1–2 components | 3 components | 4 components | 5+ components |
| Code quality | Runs with errors | Runs on happy path | Handles errors | Production-ready |
| Evaluation | No metrics | One metric, few examples | Two metrics, 10+ examples | RAGAS or ground truth |
| Architecture clarity | No diagram | Rough sketch | Clear diagram + rationale | Matches code exactly |
| Demo quality | Doesn't run | Runs once | Reliable, shows edge cases | Recorded + narrated |

---

> [!success] The most important thing
> A working system that does one thing well is more valuable than a half-built system that claims to do five things. Ship the core pipeline first, then add features.

---

[[05-presentation-guide]]
