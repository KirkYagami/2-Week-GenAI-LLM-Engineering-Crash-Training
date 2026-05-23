# Agenda — LLM Evaluation

Shipping an LLM feature without evaluation is shipping blind. You need to measure what you can't see: faithfulness, relevance, hallucination rate, and whether your prompt changes actually improved anything.

## Learning objectives

By the end of this session you will be able to:

- Design an evaluation strategy for LLM outputs
- Use RAGAS to evaluate RAG pipeline quality
- Implement LLM-as-judge for automated scoring
- Build a human evaluation framework with calibrated rubrics

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:25 | Evaluation overview — why evals, types, design | [[01-evaluation-overview]] |
| 0:25 – 1:00 | RAGAS framework — faithfulness, relevance, recall | [[02-ragas-framework]] |
| 1:00 – 1:30 | Hallucination and faithfulness metrics | [[03-hallucination-and-faithfulness]] |
| 1:30 – 2:00 | Relevance metrics — answer, context, query | [[04-relevance-metrics]] |
| 2:00 – 2:30 | Human evaluations — rubrics, inter-rater, calibration | [[05-human-evals]] |
| 2:30 – 3:00 | Practice exercises | [[06-practice-exercises]] |

## Setup

```bash
pip install ragas openai anthropic datasets pandas
```

```python
import os
from openai import OpenAI
from anthropic import Anthropic

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

[[../Day-03-Part-2-Vector-Databases/08-interview-questions|← Day 03 Part 2]] | [[01-evaluation-overview|Start →]]
