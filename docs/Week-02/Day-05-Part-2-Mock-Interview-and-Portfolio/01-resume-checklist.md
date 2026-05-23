# Resume Checklist for LLM Engineering Roles

A resume for LLM engineering roles is different from a general software engineering resume: recruiters and hiring managers are looking for specific signals that you have built real systems with real AI components, not just completed tutorials.

## Learning objectives

- Write resume bullets that quantify LLM project work
- Identify and list the skills that LLM engineering job postings actually screen for
- Avoid the most common resume mistakes for AI/ML roles

---

## Skills section — what to include

List these only if you can speak to them in an interview. A skill on your resume is an invitation to be asked about it.

### Core LLM skills (high signal)
- `OpenAI API`, `Anthropic API` — not just "LLMs"
- `RAG (Retrieval-Augmented Generation)` with the specific stack: `ChromaDB`, `Pinecone`, `Qdrant`
- `LangChain`, `LangGraph` — name the version era you know (LCEL, not legacy)
- `Function calling / tool use` — OpenAI tools API, Anthropic tool_use
- `Prompt engineering` — few-shot, chain-of-thought, structured output
- `Fine-tuning` — LoRA, QLoRA, PEFT, `transformers`, `SFTTrainer`
- `Streaming` — SSE, FastAPI `StreamingResponse`
- `LLMOps` — LangSmith, cost tracking, latency optimization

### Deployment and infrastructure (expected)
- `FastAPI`, `uvicorn`, `Pydantic`
- `Docker`, `Modal`, `Fly.io` (whichever you've used)
- `async/await`, `asyncio` — important signal for LLM services

### Evaluation (differentiator — most candidates skip this)
- `RAGAS` — faithfulness, answer relevancy, context precision
- `LangSmith` — tracing, datasets, evaluators
- Metrics: latency P95, cache hit rate, token cost per query

---

## How to write resume bullets for LLM projects

The formula: **Action verb + what you built + technology used + measurable outcome**

### Bad examples
- "Built a chatbot using LangChain and OpenAI"
- "Implemented RAG pipeline"
- "Used LLMs to build a question answering system"

### Good examples
- "Built a RAG customer support bot with FastAPI and ChromaDB that reduced average answer latency from 4.2s to 380ms via semantic caching, handling 200 req/day"
- "Implemented a LangGraph multi-agent research pipeline with planner → researcher → critic loop; achieved 0.82 average quality score across 20 test queries"
- "Deployed a document extraction API using OpenAI function calling and Pydantic validation; achieved 84% field-level accuracy on a 15-document test set"
- "Reduced LLM API costs by 40% using prompt caching and exact-match response caching on deterministic queries"
- "Designed and evaluated a QLoRA fine-tuning pipeline for sentiment classification; improved F1 from 0.71 (zero-shot) to 0.89 on held-out test set"

---

## What to include for each project

For each project in your portfolio, your resume bullet should answer:

1. What problem did it solve?
2. What technologies did it use?
3. What was a measurable outcome?

```
✓ "Built a FastAPI RAG service (ChromaDB + gpt-4o-mini) for customer Q&A;
   evaluated 20 test questions, 78% keyword precision, P95 latency 1.8s"

✗ "Created a question answering chatbot"
```

---

## Common resume mistakes for AI/ML roles

**Listing every library you've imported.** If you have `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `pytorch`, `transformers`, `langchain`, `openai` all in the skills section, it signals you've touched all of them but mastered none. Be selective.

**No metrics.** "Improved performance" is not a metric. "Reduced P95 latency from 3.1s to 0.9s with a caching layer" is.

**Vague project descriptions.** "Worked on LLM projects at [Company]" tells a hiring manager nothing. Name the system, name the technology, name the outcome.

**Listing GPT-4 experience from the ChatGPT UI.** Using the ChatGPT web interface is not the same as building with the OpenAI API. Be accurate.

**Hiding evaluation.** The fact that you ran a quantitative evaluation is a differentiator. Most candidates only demo. Put the number in the bullet.

> [!tip] One project with three metrics beats three projects with no metrics
> Depth signals expertise. A capstone project with measured latency, cache hit rate, and answer quality is more credible than five tutorial projects with no outcomes.

---

## Resume structure for junior/mid LLM engineering roles

```
[Name] | [Email] | [GitHub] | [LinkedIn]

## Summary (2–3 sentences)
LLM engineer with hands-on experience building RAG pipelines,
LangGraph agents, and production FastAPI services. Comfortable
with the full stack: prompt engineering → API integration →
async deployment → quantitative evaluation.

## Skills
LLM APIs: OpenAI (gpt-4o-mini, embeddings), Anthropic (claude-haiku-4-5)
Frameworks: LangChain (LCEL), LangGraph, PEFT
Infrastructure: FastAPI, Docker, Modal, ChromaDB, Pinecone
Evaluation: RAGAS, LangSmith, custom metrics
Languages: Python (primary), SQL

## Projects
[Capstone] RAG Customer Support Bot ...
[Project 2] LangGraph Research Agent ...

## Experience
...

## Education
...
```

---

[[00-agenda]] | [[02-portfolio-and-github]]
