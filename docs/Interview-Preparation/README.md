# Interview Preparation — Overview

This section prepares you for technical interviews at companies hiring LLM engineers, AI product engineers, and ML platform engineers.

## What to expect

Most LLM engineering technical screens have 4 parts:

1. **Background and projects** (10–15 min) — walk through your work, explain design decisions
2. **Technical Q&A** (20–30 min) — LLM fundamentals, RAG, agents, deployment
3. **System design** (15–25 min) — design an LLM system at scale
4. **Coding** (15–20 min) — write a function involving LLM API calls, async, or data processing

## Study resources in this section

The Week 2 Day 5 mock interview section has the most practical preparation:

- [Technical Q&A — 15 questions with model answers](../Week-02/Day-05-Part-2-Mock-Interview-and-Portfolio/03-technical-interview-questions.md)
- [System design practice — 3 problems with sample designs](../Week-02/Day-05-Part-2-Mock-Interview-and-Portfolio/04-system-design-practice.md)
- [Full mock interview script — 45-minute simulation](../Week-02/Day-05-Part-2-Mock-Interview-and-Portfolio/05-mock-interview-script.md)
- [Resume and portfolio checklist](../Week-02/Day-05-Part-2-Mock-Interview-and-Portfolio/01-resume-checklist.md)

## Interview question topics by frequency

Based on what companies actually ask:

### High frequency (study these first)
- How does RAG work? When would you use it vs fine-tuning?
- Explain the function calling cycle (OpenAI tools API)
- What is temperature and when do you use temperature=0?
- How do you handle rate limits in production?
- Why use AsyncOpenAI instead of the sync client in FastAPI?
- What is LoRA and why is it memory-efficient?

### Medium frequency
- HyDE, reranking, multi-query retrieval
- LangGraph nodes, edges, conditional routing
- RAGAS metrics: faithfulness, answer relevancy
- Exact-match vs semantic caching tradeoffs
- Serverless cold starts for LLM apps

### System design (one of these will appear)
- Design a customer support bot backed by documentation
- Design a batch document processing pipeline (1M docs/day)
- Design a coding assistant for an internal codebase
- Design an LLM evaluation pipeline

## [[08-mock-interview-simulator]]

The mock interview simulator contains an extended set of practice questions with hints and model answers.

---

## 30-day study plan

| Week | Focus |
|------|-------|
| Week 1 | Complete the course content; build 2 projects |
| Week 2 | Practice technical Q&A out loud; build 2 more projects |
| Week 3 | System design practice (2 problems/day); polish portfolio |
| Week 4 | Full mock interviews; apply aggressively |
