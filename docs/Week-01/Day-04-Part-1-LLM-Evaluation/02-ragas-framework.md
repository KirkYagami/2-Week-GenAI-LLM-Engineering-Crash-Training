# RAGAS Framework

RAGAS (Retrieval-Augmented Generation Assessment) is the standard evaluation framework for RAG pipelines. It measures four dimensions: faithfulness, answer relevancy, context precision, and context recall — without requiring human-labeled ground truth for every metric.

## Learning objectives

- Run RAGAS evaluations on a RAG pipeline output
- Interpret each metric and what low scores indicate
- Build a custom eval dataset from scratch
- Integrate RAGAS into a CI/CD evaluation pipeline

---

## RAGAS metrics

| Metric | Measures | Requires ground truth? | Score range |
|--------|---------|----------------------|-------------|
| **Faithfulness** | Does the answer contain only information from the context? | No | 0–1 |
| **Answer Relevancy** | Does the answer actually respond to the question? | No | 0–1 |
| **Context Precision** | Are retrieved contexts relevant to the question? | Yes (ground truth context) | 0–1 |
| **Context Recall** | Does the retrieved context cover the ground truth answer? | Yes (ground truth answer) | 0–1 |

---

## Basic RAGAS evaluation

```python
import os
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# RAGAS expects a dataset with these columns:
# - question: the user query
# - answer: the generated answer
# - contexts: list of retrieved text chunks
# - ground_truth: the reference answer (required for context_precision and context_recall)

eval_data = {
    "question": [
        "What is Python used for?",
        "How does RAG reduce hallucination?",
        "What is cosine similarity?"
    ],
    "answer": [
        "Python is used for data science, web development, automation, and AI.",
        "RAG reduces hallucination by injecting relevant documents into the prompt so the model answers from retrieved text rather than from memory.",
        "Cosine similarity measures the angle between two vectors. It returns 1.0 for identical vectors and 0.0 for orthogonal ones."
    ],
    "contexts": [
        ["Python is a high-level programming language used for data science, ML, and web development.",
         "Python's simple syntax makes it popular for automation and scripting."],
        ["RAG combines retrieval with generation. Retrieved documents are injected into the prompt.",
         "Language models hallucinate when they don't know the answer — RAG grounds responses in retrieved text."],
        ["Cosine similarity measures the cosine of the angle between two vectors.",
         "For normalized vectors, cosine similarity equals the dot product."]
    ],
    "ground_truth": [
        "Python is used for data science, machine learning, web development, and automation.",
        "RAG reduces hallucination by providing relevant documents in the context window, so the model reads the answer instead of recalling it.",
        "Cosine similarity is a metric that measures the angle between two vectors, returning values from -1 to 1."
    ]
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=None  # uses gpt-3.5-turbo by default; set to your preferred model
)

print(results)
print(f"\nFaithfulness:     {results['faithfulness']:.3f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
print(f"Context Precision:{results['context_precision']:.3f}")
print(f"Context Recall:   {results['context_recall']:.3f}")
```

---

## RAGAS with a custom LLM judge

Configure RAGAS to use a specific OpenAI model for scoring:

```python
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

# Use gpt-4o as the judge model (more accurate than default gpt-3.5)
judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
)

# Use text-embedding-3-small for semantic similarity
embedding_model = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=embedding_model
)
print(results)
```

---

## Interpreting RAGAS scores

```python
def interpret_ragas_scores(scores: dict) -> str:
    thresholds = {
        "faithfulness": (0.8, "Critical — model is adding information not in context"),
        "answer_relevancy": (0.7, "Important — model is not directly answering the question"),
        "context_precision": (0.7, "Important — irrelevant chunks are being retrieved"),
        "context_recall": (0.7, "Important — relevant documents are not being retrieved")
    }

    issues = []
    for metric, (threshold, message) in thresholds.items():
        score = scores.get(metric)
        if score is not None and score < threshold:
            issues.append(f"  ⚠ {metric} = {score:.3f} (< {threshold}): {message}")

    if not issues:
        return "All metrics within acceptable range."
    return "Issues found:\n" + "\n".join(issues)

# Example interpretation
scores = {
    "faithfulness": 0.72,
    "answer_relevancy": 0.85,
    "context_precision": 0.65,
    "context_recall": 0.78
}
print(interpret_ragas_scores(scores))
# Output:
# Issues found:
#   ⚠ faithfulness = 0.720 (< 0.800): Critical — model is adding information not in context
#   ⚠ context_precision = 0.650 (< 0.700): Important — irrelevant chunks are being retrieved
```

**Diagnosis guide:**

| Low metric | Root cause | Fix |
|-----------|-----------|-----|
| Low faithfulness | Model using parametric knowledge | Strengthen "use only context" instruction |
| Low answer relevancy | Model answering the wrong question | Improve question understanding, add clarification step |
| Low context precision | Too many irrelevant chunks retrieved | Better embedding model, reduce k, add reranking |
| Low context recall | Missing relevant documents | Better chunking, add query expansion, hybrid search |

---

## Building a RAGAS evaluation set from your corpus

```python
def generate_ragas_examples(
    documents: list[dict],
    n_questions: int = 20
) -> list[dict]:
    """Auto-generate RAGAS evaluation examples from your documents using GPT-4o."""
    from openai import OpenAI
    import json

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    examples = []

    for doc in documents[:n_questions]:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Given this document, generate a question and ideal answer.
Return JSON with:
- "question": a natural question someone might ask
- "ground_truth": the ideal concise answer based only on the document

Document: {doc['text']}

JSON only:"""
            }],
            max_tokens=200,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        qa = json.loads(response.choices[0].message.content)
        examples.append({
            "question": qa["question"],
            "ground_truth": qa["ground_truth"],
            "source_doc_id": doc["id"]
        })

    return examples

# Usage with your document corpus
corpus = [
    {"id": "1", "text": "Python was created by Guido van Rossum and first released in 1991."},
    {"id": "2", "text": "The transformer architecture was introduced in 2017 in the paper 'Attention Is All You Need'."},
    {"id": "3", "text": "RAG stands for Retrieval-Augmented Generation, introduced by Meta AI in 2020."},
]

# generated_examples = generate_ragas_examples(corpus, n_questions=3)
# for ex in generated_examples:
#     print(f"Q: {ex['question']}\nA: {ex['ground_truth']}\n")
```

---

## RAGAS in a CI/CD pipeline

```python
import subprocess
import json

def run_ragas_eval(rag_pipeline, eval_dataset_path: str) -> dict:
    """Run RAGAS evaluation and fail if below threshold."""
    with open(eval_dataset_path) as f:
        eval_data = json.load(f)

    # Run your RAG pipeline on each question
    answers = []
    contexts_list = []
    for item in eval_data:
        result = rag_pipeline.ask(item["question"])
        answers.append(result["answer"])
        contexts_list.append(result.get("retrieved_texts", ["No context"]))

    dataset = Dataset.from_dict({
        "question": [item["question"] for item in eval_data],
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": [item.get("ground_truth", "") for item in eval_data]
    })

    scores = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])
    scores_dict = dict(scores)

    THRESHOLDS = {
        "faithfulness": 0.75,
        "answer_relevancy": 0.70
    }

    failures = {
        metric: score
        for metric, score in scores_dict.items()
        if metric in THRESHOLDS and score < THRESHOLDS[metric]
    }

    if failures:
        print(f"EVAL FAILED: {failures}")
        # In CI: exit(1)
    else:
        print(f"EVAL PASSED: {scores_dict}")

    return {"scores": scores_dict, "passed": len(failures) == 0, "failures": failures}

# In GitHub Actions:
# python -c "from eval import run_ragas_eval; run_ragas_eval(pipeline, 'eval_set.json')"
```

---

[[01-evaluation-overview]] | [[03-hallucination-and-faithfulness]]
