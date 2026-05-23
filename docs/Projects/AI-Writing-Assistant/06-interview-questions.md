# Interview Questions — AI Writing Assistant

---

**Q1: You have 4 sequential LLM calls in this pipeline. What's the total latency and how would you reduce it?**

??? "Show answer"
    Each gpt-4o-mini call for a 300-400 token output takes 800–1,500ms. Four sequential calls = 3.2–6s total. To reduce it:
    
    1. **Parallelize independent steps.** The outline and a "style analysis" step (if added) could run in parallel since the style analysis doesn't depend on the outline. Use `asyncio.gather()`.
    
    2. **Skip steps for simple requests.** A short casual blog post probably doesn't need a separate refinement pass. Add a `fast_mode: bool` option that skips refine and style steps.
    
    3. **Reduce max_tokens per step.** Each step has a hard ceiling; reducing it speeds up generation when the model would naturally stop before the limit anyway.
    
    4. **Stream the final step.** Stream the style step tokens to the client instead of waiting for the full completion — users see output starting in 800ms instead of after 6s.

---

**Q2: Why do you use LangChain's LCEL instead of calling the OpenAI API directly?**

??? "Show answer"
    LCEL gives you composability (chain operators `|`), built-in async support (`.ainvoke()`, `.astream()`), and output parsers that handle parsing and error propagation. For a 4-step pipeline, writing the equivalent with raw API calls requires explicit `await` on each call, manual error handling between steps, and custom streaming logic.
    
    The tradeoff: LCEL adds a dependency and its own abstraction layer. For simple single-step calls, raw `AsyncOpenAI` is cleaner. For multi-step pipelines with conditional logic, LCEL's composability pays for itself.

---

**Q3: How would you handle a request where the LLM generates a draft that's 2x longer than requested?**

??? "Show answer"
    The refine step can include an explicit length constraint: `"Reduce to approximately {target_words} words while preserving all key points."` 
    
    For more control: after the draft step, count words and only invoke the refine step with a length reduction instruction if the draft is >20% over target. This avoids a refine step when the draft is on-target.
    
    As a last resort: add a post-processing truncation at sentence boundaries. This is a hack — it breaks the coherence of the ending — so prompt-level length control is always preferable.

---

**Q4: How would you add user authentication so each user has their own usage quota?**

??? "Show answer"
    Add an `Authorization: Bearer <token>` header. Validate the token against a database (or a simple hardcoded dict for a prototype). Attach a `user_id` to each request. Track usage per `user_id` in a Redis sorted set (ZADD with timestamp score, ZRANGEBYSCORE to count recent requests).
    
    ```python
    from fastapi import Header, HTTPException
    
    async def get_current_user(authorization: str = Header(None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing token")
        token = authorization.split(" ")[1]
        user_id = validate_token(token)  # DB lookup
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    ```

---

[[05-deployment]]
