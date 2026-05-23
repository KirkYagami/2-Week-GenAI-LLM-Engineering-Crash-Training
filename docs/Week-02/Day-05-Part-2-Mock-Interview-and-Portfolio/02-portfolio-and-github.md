# Portfolio and GitHub

Your GitHub profile is the first thing a technical interviewer opens after reading your resume. A well-structured portfolio repository communicates competence before they read a single line of code.

## Learning objectives

- Structure a GitHub repository to maximize signal for technical reviewers
- Write a README that sells the project in 30 seconds
- Pin the right repositories and write a useful profile README

---

## What a strong LLM project repository looks like

### Repository structure

```
my-rag-chatbot/
├── README.md              ← First thing reviewers read — make it count
├── DESIGN.md              ← Architecture decisions and tradeoffs
├── requirements.txt       ← Pinned versions
├── .env.example           ← Required env vars with blank values
├── .gitignore             ← Must include .env, __pycache__, *.pyc, chroma_db/
├── app.py                 ← Main application
├── ingest.py              ← Data pipeline
├── eval.py                ← Evaluation script with results
├── docs/                  ← Source documents for RAG
└── screenshots/           ← Demo screenshot or architecture diagram image
```

### .gitignore (critical — never commit secrets)

```gitignore
.env
*.pyc
__pycache__/
chroma_db/
.DS_Store
*.egg-info/
dist/
build/
```

---

## README structure that converts

A reviewer opens your README and decides in 30 seconds whether to keep reading. Structure it so the most important information is at the top.

```markdown
# RAG Customer Support Bot

A streaming FastAPI service that answers customer questions by retrieving
from a documentation corpus, with exact-match caching and LangSmith tracing.

## Demo

[Screenshot or GIF of the running application]

## Architecture

[Mermaid diagram or link to DESIGN.md]

User question → Cache check → ChromaDB retrieval → gpt-4o-mini → SSE stream

## Evaluation results

| Metric | Value |
|--------|-------|
| Avg keyword precision (20 questions) | 78% |
| P95 latency | 1.8s |
| Cache hit rate (repeated queries) | 100% |

## Quick start

\```bash
git clone https://github.com/yourname/rag-chatbot
cd rag-chatbot
cp .env.example .env   # Add your OPENAI_API_KEY
pip install -r requirements.txt
python ingest.py       # Index documents
uvicorn app:app --reload
\```

## Tech stack

- **LLM:** OpenAI gpt-4o-mini (text generation), text-embedding-3-small (retrieval)
- **Vector DB:** ChromaDB (local persistent)
- **API:** FastAPI with async streaming (SSE)
- **Caching:** Exact-match SHA256 cache with 10-minute TTL
- **Tracing:** LangSmith

## Key design decisions

- **gpt-4o-mini over gpt-4o:** 10x cheaper, acceptable quality for factual Q&A
- **Exact-match cache before embedding:** Avoids embedding cost on repeated queries
- **Top-5 retrieval:** Empirically better than top-3 (misses answers) or top-10 (dilutes context)
```

---

## Profile README

Create a file at `github.com/yourusername/yourusername/README.md` — GitHub renders it on your profile page.

```markdown
## Hi, I'm [Name]

LLM engineer building production AI systems with FastAPI, LangGraph, and RAG.

### Current projects
- [RAG Customer Support Bot](link) — streaming FastAPI service, 78% precision on 20-question eval
- [LangGraph Research Agent](link) — multi-node graph with critic-rewrite loop

### Skills
Python · FastAPI · OpenAI API · Anthropic API · LangChain/LangGraph · ChromaDB · RAGAS · Docker

### Connect
[LinkedIn](link) · [Email](link)
```

---

## Pinned repositories

Pin exactly 6 repositories. Pick them in this order of priority:

1. **Capstone project** — your most complete, evaluated, deployed work
2. **One agent project** — shows LangGraph or ReAct agent implementation
3. **One RAG project** — shows retrieval pipeline design
4. **One fine-tuning project** — if you have one (strong differentiator)
5. **One utility/tool** — a smaller, clean, useful library or script
6. **One non-LLM project** — shows you have a software engineering foundation

If you don't have 6, don't pin placeholder repos. Three polished pinned repos are better than six half-finished ones.

---

## Commit history signals

Technical reviewers sometimes look at commit history. These signals matter:

**Good:**
- Regular commits over time, not one giant commit the night before the demo
- Descriptive commit messages: `add semantic caching with cosine similarity threshold`, not `update files`
- Commits that show iteration: you added a feature, measured it, adjusted it

**Bad:**
- `initial commit` containing 2,000 lines of code
- `fix stuff` as a commit message
- Committing `.env` files (even if you delete them later, they're in the history)

> [!warning] Secrets in git history are permanent
> Even if you delete a committed `.env` file, the secret is in the git history and can be extracted. If you accidentally commit a key, rotate it immediately and use `git filter-branch` or BFG Repo Cleaner to remove it from history. Then force-push and notify anyone with a copy of the repo.

---

## Portfolio checklist

- [ ] Capstone repository has README with demo, architecture, eval results, and setup instructions
- [ ] `.gitignore` includes `.env` and credential files
- [ ] `.env.example` included with blank values
- [ ] No hardcoded API keys anywhere in commit history
- [ ] GitHub profile has a profile README
- [ ] At least 2 repos pinned on profile
- [ ] Each project has at least one quantitative result in the README

---

[[01-resume-checklist]] | [[03-technical-interview-questions]]
