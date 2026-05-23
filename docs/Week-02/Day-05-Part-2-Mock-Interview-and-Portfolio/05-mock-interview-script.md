# Mock Interview Script

This is a complete mock interview covering a 45-minute technical screen for a junior LLM engineer role. Run it in pairs: one person reads the interviewer questions, the other answers. After each answer, the interviewer reads the feedback note before moving on.

---

## Setup instructions

**Interviewer:** Read each question exactly as written. Give the candidate 2–3 minutes per question. Use the feedback note to calibrate — don't read it aloud until the candidate has finished answering.

**Candidate:** Answer as if this is a real interview. No notes. Speak your reasoning out loud. If you don't know something, say what you do know and where your knowledge ends.

**Time:** Allow 45 minutes total. Use the time markers as guides.

---

## Part 1 — Introduction (5 minutes)

**Q1 (Interviewer):** "Tell me about yourself and what drew you to LLM engineering."

> [!info] What to listen for
> A confident answer covering: background, what they've built (not just studied), and a specific thing about LLMs that interests them technically. Generic answers ("I'm passionate about AI") are weak. Strong: "I built a RAG pipeline that..., and I got interested in how retrieval quality affects answer faithfulness."

---

**Q2 (Interviewer):** "Walk me through the most complex LLM project you've built. I want to understand the architecture, the challenges, and what you'd do differently."

> [!info] What to listen for
> Specificity: model names, library versions, how they measured success. Does the candidate understand their own design decisions? Do they have an honest answer about what didn't work? A candidate who says "everything worked great" has either never shipped or isn't self-aware.

---

## Part 2 — Technical depth (25 minutes)

**Q3 (Interviewer):** "You're building a RAG service. The user asks: 'What is our refund policy?' Your retrieval returns three chunks about refund policy — but two are from an old version of the policy that's been superseded. How do you handle this?"

> [!info] What to listen for
> The candidate should identify: (1) the metadata problem — you need document versioning and filtering; (2) the solution — store `version` or `last_updated` as ChromaDB metadata and filter by date; (3) the deeper issue — retrieval quality depends on index freshness. A great answer also mentions: monitoring for staleness, re-indexing on document update, and potentially using the document date in the prompt to let the model reason about which chunk is current.

---

**Q4 (Interviewer):** "Explain what happens at the code level when you make a streaming API call to OpenAI, and how SSE works end-to-end from the server to the browser."

> [!info] What to listen for
> Level of detail: the candidate should cover (1) `stream=True` in the API call; (2) the generator pattern — `async for chunk in response`; (3) SSE format: `data: {json}\n\n`; (4) `StreamingResponse` in FastAPI; (5) `X-Accel-Buffering: no` for nginx; (6) the client reading `EventSource` or `iter_lines()`. Missing the proxy buffering issue is a common gap.

---

**Q5 (Interviewer):** "I'm seeing that our LLM service's P95 latency is 6 seconds, but P50 is 800ms. What's causing this, and what would you do to diagnose and fix it?"

> [!info] What to listen for
> Diagnosis: P95 vs P50 gap suggests occasional slow requests, not uniformly slow service. Causes: long prompts, slow retrieval for certain queries, LLM API rate limiting causing queue buildup, cold starts for serverless. The candidate should want to look at: latency breakdown by component (retrieval vs LLM), the distribution of prompt token counts, whether the P95 requests are retries. Fixes: prompt length cap, cache warming, min_containers=1 for serverless, async retry logic.

---

**Q6 (Interviewer):** "When would you choose to fine-tune a model versus use RAG? Give me a scenario where each is clearly the right choice."

> [!info] What to listen for
> RAG scenario: a customer support bot for a company with a documentation base that changes weekly. Fine-tuning scenario: a company that needs all LLM outputs to follow a specific JSON schema, uses the same format 10,000 times per day, and has labeled training data. The candidate should know the two key questions: does the knowledge change? Do I have labeled data? A strong answer also mentions: RAG is better for knowledge, fine-tuning is better for behavior/style.

---

**Q7 (Interviewer):** "Write me a Python function that makes a call to the OpenAI chat API with automatic retry logic for rate limits. Use exponential backoff."

*Give the candidate a whiteboard or text editor.*

```python
# Expected answer (or close to it):
import asyncio
import random
from openai import AsyncOpenAI, RateLimitError

aclient = AsyncOpenAI()

async def chat_with_retry(messages: list[dict], max_retries: int = 3) -> str:
    base_delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            response = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
```

> [!info] What to listen for
> (1) Uses `AsyncOpenAI`, not `OpenAI`; (2) catches `RateLimitError` specifically, not bare `Exception`; (3) re-raises on the last attempt; (4) adds jitter to prevent thundering herd; (5) uses `await asyncio.sleep` not `time.sleep` (which would block). The note about the OpenAI SDK having built-in retry is a bonus signal.

---

## Part 3 — System design (10 minutes)

**Q8 (Interviewer):** "Design the backend for a document Q&A product. Users upload PDFs and can ask questions about them. There are 1,000 users, each with 10–100 PDFs, each PDF is 5–50 pages."

*Give the candidate a whiteboard or text editor. Expect them to spend 1–2 minutes asking clarifying questions.*

> [!info] What to listen for
> Clarifying questions: concurrent users? latency requirement? multi-user or single-user isolation? Key design decisions: per-user collection isolation in ChromaDB/Pinecone (security boundary); chunking strategy for PDFs (page-level vs paragraph-level); async ingestion (don't block the upload endpoint on embedding); streaming for Q&A. Failure modes: what if the PDF is image-only? (OCR or graceful error). The candidate should be able to sketch: upload API → async ingestion worker → vector store → Q&A API → RAG pipeline.

---

## Part 4 — Behavioral (5 minutes)

**Q9 (Interviewer):** "Tell me about a time an LLM gave you an unexpected output that caused a problem. What did you learn?"

> [!info] What to listen for
> Honesty and specificity. "LLMs sometimes hallucinate" is not an answer. A strong answer: "I had a function-calling pipeline where the model occasionally returned JSON with a key in a different case (`Total_Amount` vs `total_amount`) which broke my Pydantic validation. I learned to normalize extracted keys and add strict schema validation with error logging."

---

**Q10 (Interviewer):** "If you had two weeks to improve a RAG system's answer quality from 70% to 90% on a test set, what would you do, in what order?"

> [!info] What to listen for
> The answer should prioritize high-leverage changes first: (1) analyze failure cases — are they retrieval failures or generation failures? (2) fix retrieval first — better chunking, reranker; (3) fix generation — better prompt, more context, lower temperature; (4) add semantic caching to avoid re-testing the same queries. A good answer establishes the diagnosis step before prescribing solutions.

---

## Debrief template

After the mock, give the candidate feedback on each dimension:

| Dimension | Observation | Suggestion |
|-----------|-------------|------------|
| Technical accuracy | | |
| Specificity (named real tools/models) | | |
| Tradeoff reasoning | | |
| Communication clarity | | |
| Handling of unknown questions | | |
| Coding exercise | | |

**Overall:** What is the single most impactful thing this candidate could do to improve their interview performance in the next two weeks?

---

[[04-system-design-practice]]
