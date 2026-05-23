# Interview Questions — Document Summarizer

---

**Q1: Why use map-reduce instead of just truncating the document to fit the context window?**

??? "Show answer"
    Truncation discards information — if the answer to a question is in page 15 of a 20-page report and you truncate at page 10, you've lost it. Map-reduce preserves all content: every chunk is summarized, and the reduce step synthesizes across all chunk summaries. The tradeoff is cost and latency — map-reduce makes one LLM call per chunk plus one reduce call, versus one call for a truncated prompt. For a 20-chunk document, that's 21 calls instead of 1. The quality improvement is worth it for cases where completeness matters.

---

**Q2: What is the risk of overlap between chunks and how do you balance it?**

??? "Show answer"
    Overlap (200 tokens in this project) means the same sentence appears in two chunks, leading to that information being summarized twice. In the reduce step, the model may include duplicate points in the final summary. The benefit: no sentence is lost at a chunk boundary, so answers that cross a boundary are captured.
    
    Balance: 10–20% overlap is typical. More overlap = more duplication in intermediate summaries. Less overlap = risk of losing information at boundaries. For narrative text (articles, reports), 15% overlap works well. For structured text (financial reports with discrete sections), you can use semantic chunking at section boundaries with zero overlap.

---

**Q3: How would you evaluate whether the executive summary format is being followed?**

??? "Show answer"
    Three levels of checking:
    
    1. **Structural check** — does the output contain a TL;DR line? Do bullet points start with `-` or `•`? This is fast and rule-based.
    
    2. **LLM-as-judge** — prompt a judge model: "Does this text follow the executive summary format (1-sentence TL;DR followed by 3–5 key takeaways as bullet points)? Answer yes/no." Parse the response.
    
    3. **Pydantic validation** — for structured outputs, use `with_structured_output(ExecutiveSummarySchema)` to enforce the schema at the API level. Any response that doesn't parse to the schema triggers a retry.

---

**Q4: What happens if the reduce step's input (all chunk summaries combined) is longer than the context window?**

??? "Show answer"
    For very long documents, the chunk summaries themselves can exceed the context window when combined. This is why the advanced features section shows recursive summarization: if the combined chunk summaries are too long, treat them as a new document and run another map-reduce pass. This adds latency (roughly proportional to the number of recursive levels) but handles arbitrarily long documents.
    
    A practical guard: check `count_tokens(combined_summaries)` before the reduce call and recursively summarize if it exceeds `max_context - 600` (leaving room for the system prompt and output).

---

[[05-deployment]]
