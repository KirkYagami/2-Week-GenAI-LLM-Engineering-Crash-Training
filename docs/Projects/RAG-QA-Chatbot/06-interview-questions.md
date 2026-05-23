# Interview Questions — RAG Q&A Chatbot

These questions are specifically about the design decisions in this project. Prepare concrete answers citing your implementation choices.

---

**Q1: How did you chunk the documents and why?**

??? "Show answer"
    Fixed-size chunks of 800 characters with a 150-character overlap. The overlap prevents answers from being split across chunk boundaries — any sentence that spans two chunks appears in both, so retrieval won't miss it. Fixed-size is simpler to implement and reason about than semantic chunking; for a portfolio project with well-structured docs, it performs acceptably. For production with dense technical PDFs, paragraph-level semantic chunking (splitting at double newlines, respecting section headers) would improve precision.

---

**Q2: Why gpt-4o-mini instead of gpt-4o for this project?**

??? "Show answer"
    This is a factual Q&A task: the answer is grounded in retrieved context, so the model's role is reading comprehension and citation, not open-ended reasoning. gpt-4o-mini handles this well at 10x lower cost. gpt-4o would be appropriate if: the retrieved context is long and complex (requiring multi-step reasoning to synthesize), the questions involve ambiguity that requires careful interpretation, or quality evaluations showed gpt-4o-mini failing on specific question types.

---

**Q3: How does your caching work, and what inputs determine the cache key?**

??? "Show answer"
    SHA-256 hash of a JSON-serialized object containing the question and the `n_results` parameter. Both are included because the same question with a different number of retrieved chunks could produce a different answer (more chunks = more context = potentially longer/different answer). Temperature is fixed at 0.0 for all cacheable requests — caching non-deterministic outputs would serve stale responses. The TTL is 10 minutes; after that, the cache entry expires and the next request goes to the LLM.

---

**Q4: What are the failure modes of this system and how do you handle them?**

??? "Show answer"
    Three main failure modes:
    
    1. **Retrieval failure** — the question is about something not in the corpus. Handled by checking `len(chunks) == 0` and returning a 404 with a clear message rather than generating a hallucinated answer.
    
    2. **LLM API failure** — OpenAI returns 429 (rate limit) or 5xx. The AsyncOpenAI client has built-in exponential backoff for transient errors. For sustained outages, the appropriate response is to surface the error to the caller, not serve a cached (potentially stale) response.
    
    3. **Stale index** — the documents were updated but the index wasn't re-ingested. The freshest content is in the docs directory but the index still has old chunks. Mitigation: run `ingest.py` on every document update, or add an incremental re-index that compares file modification times.

---

**Q5: How would you scale this from 100 to 10,000 requests/day?**

??? "Show answer"
    The current architecture is single-instance with an in-memory cache. At 10,000 requests/day, the bottlenecks are:
    
    - **In-memory cache**: doesn't persist across restarts and doesn't work with multiple instances. Replace with Redis.
    - **ChromaDB**: a local PersistentClient is fine for one instance. At scale with concurrent writes (re-ingestion during traffic), move to Pinecone or a hosted ChromaDB.
    - **Single FastAPI instance**: use Fly.io's auto-scaling or deploy multiple replicas behind a load balancer. Async FastAPI handles concurrent requests well, so one instance with a proper server (uvicorn with multiple workers) can handle significant load before needing horizontal scaling.
    
    The OpenAI API is not a bottleneck unless you're exceeding your tier's TPM limit — in which case, add a semaphore to limit concurrent LLM calls or upgrade the tier.

---

[[05-deployment]]
