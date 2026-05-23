# Week 01 Assignment

Complete this assignment before starting Week 2. It covers the five core skills from Week 1.

---

## Submission format

Create a GitHub repository named `llm-week01` with the following structure:

```
llm-week01/
├── README.md      ← Summary of what you built and your results
├── task1.py
├── task2.py
├── task3.py
├── task4.py
├── task5.py
└── requirements.txt
```

Each script should run with `python taskN.py` and print its results.

---

## Task 1 — API fundamentals (2 points)

Write a Python script that:
- Calls the OpenAI chat API to answer 5 different questions
- Measures and prints the latency for each call
- Prints the total token count and estimated cost for all 5 calls combined

**Expected output:**
```
Q1: What is Python? → Answer in 10 words... | 45 tokens | 312ms
Q2: ...
...
Total: 210 tokens | $0.000032 | avg 350ms
```

---

## Task 2 — Embeddings and similarity (2 points)

Write a script that:
- Takes 10 sentences (your choice of topic)
- Embeds all 10 using `text-embedding-3-small`
- Finds and prints the top-3 most similar pairs using cosine similarity
- Finds and prints the most dissimilar pair

**Expected output:**
```
Most similar pairs:
  1. "Python is a programming language" ↔ "Python is used for data science" — similarity: 0.943
  2. ...
Most dissimilar pair:
  "Python is a programming language" ↔ "The sky is blue" — similarity: 0.121
```

---

## Task 3 — Basic RAG (3 points)

Build a minimal RAG system:
- Ingest at least 5 markdown files (or 1 PDF) into ChromaDB
- Accept a question from the user (use `input()`)
- Retrieve the top-3 relevant chunks
- Generate an answer with citations
- Print the answer and which source(s) it came from

Run it end-to-end and screenshot or paste the output in your README.

---

## Task 4 — Evaluation (2 points)

Using your Task 3 system:
- Define 5 test questions with expected source documents
- Run retrieval for each question
- Calculate Recall@3: fraction of test questions where the expected source appears in the top-3 results
- Print the results

**Expected output:**
```
Q1: "What is chunking?" → Sources: [faq_2, faq_5, guide_1] | Expected: faq_2 | HIT
Q2: ...
Recall@3: 4/5 = 80%
```

---

## Task 5 — Structured output (1 point)

Write a script that uses function calling (OpenAI tools API or `with_structured_output`) to extract structured data from 3 different unstructured text samples. The schema should include at least 4 fields with different types (string, int/float, list, optional).

Print each extraction result as a Pydantic model.

---

## Grading rubric

| Task | Points | Pass criteria |
|------|--------|--------------|
| 1 | 2 | All 5 calls work, latency and cost printed accurately |
| 2 | 2 | Similarity computation correct, output includes actual similarity scores |
| 3 | 3 | End-to-end pipeline works with real documents, citations present |
| 4 | 2 | Recall@3 computed correctly over ≥5 test questions |
| 5 | 1 | Extraction returns a valid Pydantic model for all 3 inputs |

Total: 10 points. Pass threshold: 7/10.

---

> [!tip] Don't optimize until it works
> Get a working end-to-end result first, then improve. A working system with 60% retrieval recall is worth more than a theoretically perfect system that doesn't run.
