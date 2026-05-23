# Presentation Guide

Five minutes is enough time to demonstrate a working system and leave an impression — if you structure it correctly. Most demo failures come from poor time management and unclear framing, not bad code.

## Learning objectives

- Structure a 5-minute technical demo that communicates what you built and why it works
- Anticipate the questions you'll get and prepare concise answers
- Present evaluation results as evidence, not decoration

---

## The 5-minute structure

| Time | What to cover |
|------|--------------|
| 0:00–0:45 | Problem statement — what problem does this solve? Who is the user? |
| 0:45–1:30 | Architecture walkthrough — draw or describe the data flow in plain language |
| 1:30–3:30 | Live demo — show the real system handling 2–3 representative inputs |
| 3:30–4:30 | Evaluation results — show the numbers |
| 4:30–5:00 | What you'd do next — one or two improvements if you had another day |

---

## How to frame your demo

### Problem statement (45 seconds)

Do not start with "So I built a RAG chatbot." Start with the user's problem:

> "Customer support teams spend 40% of their time answering questions that are already answered in their docs. This system retrieves the relevant documentation and generates a grounded answer in under two seconds."

### Architecture walkthrough (45 seconds)

Use your Mermaid diagram or a whiteboard sketch. Walk through the data flow, not the code. Name each component and say why you chose it in one phrase:

> "The query hits an exact-match cache first — that handles repeated questions at < 5ms. On a miss, it goes to ChromaDB for retrieval, then to gpt-4o-mini for generation. The response streams token by token."

### Live demo (2 minutes)

Show at least three inputs:

1. **Golden path** — a question the system handles well. Don't start here. Start with...
2. **Hard case** — a question where the answer is buried in the docs. Show it still works.
3. **Edge case** — a question the docs don't cover. Show the system gracefully saying "I don't know" rather than hallucinating.

Before the demo, have your terminal ready with the server running and your test curl commands or UI open. Nothing kills a demo faster than "let me just start the server."

### Evaluation results (1 minute)

Show the numbers from your eval suite. Use a table if you have time to prepare one. Explain what each metric means and what your result implies:

> "I ran 20 questions against a keyword precision baseline. Average precision was 78% — meaning the answer contained at least three of five expected keywords in 78% of cases. The two failures were questions where the relevant docs weren't in my corpus."

Honest failure analysis is more impressive than inflated numbers.

### What you'd do next (30 seconds)

Pick the single highest-impact improvement and say what you'd need to implement it:

> "The biggest gap is retrieval quality. My current chunking is naive fixed-size. Parent-child chunking — storing small chunks for retrieval and large chunks for context — would likely push precision from 78% to 90%+."

---

## What reviewers are looking for

| Criterion | What it means in practice |
|-----------|--------------------------|
| Working system | The demo runs live without errors |
| Integration depth | At least 4 course components wired together |
| Quantitative evaluation | Numbers, not "it seems to work well" |
| Code quality | `os.getenv()` for secrets, error handling present, async where appropriate |
| Honest tradeoffs | You can explain what you traded off and why |

---

## Common demo mistakes and how to avoid them

**Showing the code instead of the system.** Reviewers want to see what the system does, not how it's implemented. Show the running application first; open code only if asked.

**Demo inputs that always work.** The most credible demos show the system failing gracefully on a hard input. It proves you understand the system's limits.

**Skipping evaluation.** "It works on my examples" is not evaluation. If you ran the eval script and got numbers, show them — even if they're lower than you'd like.

**Spending two minutes on architecture and thirty seconds on the demo.** Invert this. The demo is the evidence; the architecture is context for understanding it.

**Presenting improvements you didn't build.** "If I had more time I would add semantic caching, a reranker, prompt optimization, and fine-tuning" sounds like a backlog, not a plan. Pick one and explain concretely why it's the highest leverage.

> [!tip] Record your terminal session before the demo
> If the live demo breaks due to network issues or an API rate limit, a clean recording of a working session saves the presentation. Use `asciinema rec` or a screen recorder. Tell the audience you're playing a recording and describe what you're seeing.

---

## Checklist before presenting

- [ ] Server is running and all dependencies are installed
- [ ] API key is set in the environment (`echo $OPENAI_API_KEY` returns a value)
- [ ] Test inputs are ready (copy-pasteable, not requiring typing live)
- [ ] Eval script has been run and results are on screen or in a file
- [ ] Architecture diagram is visible (browser tab, slide, or whiteboard)
- [ ] Fallback: recording of a working demo session

---

[[04-evaluation-and-testing]] | [[06-submission-checklist]]
