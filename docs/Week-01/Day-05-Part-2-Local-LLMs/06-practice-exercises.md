# Practice Exercises — Local LLMs

---

## Exercise 1 — Ollama speed and quality benchmark (Warm-up)

Run the same five prompts through two different models (a 3B and an 8B) and compare response quality and tokens-per-second. Build a simple benchmark harness.

```python
import time
from openai import OpenAI

# Ollama with OpenAI-compatible API
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

BENCHMARK_PROMPTS = [
    ("factual", "What is the capital of Japan?"),
    ("reasoning", "If a train travels at 60 mph for 2.5 hours, how far does it go? Show your calculation."),
    ("coding", "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes."),
    ("summarization", "Summarize the key differences between supervised and unsupervised learning in 3 bullet points."),
    ("instruction_follow", "List exactly 5 common Python exceptions and one sentence each describing when they occur. Number them 1-5."),
]

MODELS = [
    "llama3.2:3b",   # Pull with: ollama pull llama3.2:3b
    "llama3.1:8b",   # Pull with: ollama pull llama3.1:8b
]

def benchmark_prompt(model: str, prompt: str, max_tokens: int = 200) -> dict:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens
    )
    elapsed = time.perf_counter() - start

    answer = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens if response.usage else len(answer.split())

    return {
        "answer": answer,
        "elapsed_sec": elapsed,
        "output_tokens": output_tokens,
        "tokens_per_sec": output_tokens / elapsed if elapsed > 0 else 0
    }

# Run benchmarks
results = {model: {} for model in MODELS}
for category, prompt in BENCHMARK_PROMPTS:
    print(f"\n📊 {category.upper()}: {prompt[:60]}...")
    for model in MODELS:
        try:
            r = benchmark_prompt(model, prompt)
            results[model][category] = r
            print(f"  [{model:<20}] {r['tokens_per_sec']:5.1f} tok/s | {r['elapsed_sec']:.1f}s | {r['answer'][:80]}...")
        except Exception as e:
            print(f"  [{model:<20}] ERROR: {e}")

# Summary table
print("\n" + "="*70)
print(f"{'Category':<20} {'3B tok/s':>10} {'8B tok/s':>10} {'Speed diff':>12}")
print("-"*70)
for category, _ in BENCHMARK_PROMPTS:
    r3b = results.get(MODELS[0], {}).get(category, {})
    r8b = results.get(MODELS[1], {}).get(category, {})
    tps_3b = r3b.get("tokens_per_sec", 0)
    tps_8b = r8b.get("tokens_per_sec", 0)
    diff = f"{tps_3b/tps_8b:.1f}×" if tps_8b > 0 else "N/A"
    print(f"{category:<20} {tps_3b:>10.1f} {tps_8b:>10.1f} {diff:>12}")
```

**What to observe:** The 3B model should be 2–3× faster. The 8B model should produce noticeably better reasoning and instruction-following. If quality is similar, use the 3B for cost/latency benefits.

---

## Exercise 2 — Local RAG pipeline with Ollama (Main)

Build a complete RAG pipeline that uses Ollama for both embedding and generation — no API costs, no data leaving your machine.

```python
import os
import json
import numpy as np
from openai import OpenAI

# Both embedding and generation use local Ollama
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Corpus: knowledge base for a fictional software company
KNOWLEDGE_BASE = [
    {"id": "kb1", "text": "Our API supports JSON and XML response formats. Set the Accept header to control format."},
    {"id": "kb2", "text": "Rate limits: Free tier allows 100 requests/minute. Pro tier allows 10,000 requests/minute."},
    {"id": "kb3", "text": "Authentication uses Bearer tokens. Generate tokens in the dashboard under Settings > API Keys."},
    {"id": "kb4", "text": "Webhooks can be configured per endpoint. Each webhook has a secret for HMAC signature verification."},
    {"id": "kb5", "text": "The SDK is available for Python, JavaScript, Go, and Ruby. Install with: pip install our-sdk"},
    {"id": "kb6", "text": "Error code 429 means rate limit exceeded. Implement exponential backoff starting at 1 second."},
    {"id": "kb7", "text": "Data is retained for 90 days. Export your data before deletion via the Data Export endpoint."},
    {"id": "kb8", "text": "The API is versioned. Current version is v2. v1 endpoints are deprecated and will be removed Q1 2026."},
]

class LocalRAG:
    """Complete RAG pipeline using only local Ollama."""

    def __init__(
        self,
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3.1:8b"
    ):
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.doc_texts = []
        self.doc_ids = []
        self.embeddings = None

    def _embed(self, text: str) -> list[float]:
        response = ollama_client.embeddings.create(
            model=self.embed_model,
            input=text
        )
        return response.data[0].embedding

    def index(self, documents: list[dict]) -> None:
        """Index a list of {"id": str, "text": str} documents."""
        print(f"Indexing {len(documents)} documents...")
        self.doc_texts = [d["text"] for d in documents]
        self.doc_ids = [d["id"] for d in documents]
        self.embeddings = np.array([self._embed(d["text"]) for d in documents])
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)
        print("Index complete.")

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve top-k relevant documents."""
        query_emb = np.array(self._embed(query))
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = self.embeddings @ query_emb
        top_k = np.argsort(scores)[::-1][:k]

        return [
            {
                "id": self.doc_ids[i],
                "text": self.doc_texts[i],
                "score": float(scores[i])
            }
            for i in top_k
        ]

    def answer(self, question: str, k: int = 3) -> dict:
        """Retrieve context and generate answer using local LLM."""
        retrieved = self.retrieve(question, k=k)
        context = "\n".join(f"[{r['id']}] {r['text']}" for r in retrieved)

        response = ollama_client.chat.completions.create(
            model=self.gen_model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer questions using ONLY the provided context. If the answer is not in context, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.0,
            max_tokens=200
        )

        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": [r["id"] for r in retrieved],
            "top_score": retrieved[0]["score"] if retrieved else 0.0
        }

# Build and test the local RAG system
rag = LocalRAG(embed_model="nomic-embed-text", gen_model="llama3.1:8b")
rag.index(KNOWLEDGE_BASE)

TEST_QUESTIONS = [
    "How do I authenticate to the API?",
    "What happens if I exceed the rate limit?",
    "Is there a Java SDK available?",  # Answer should be: not in context
    "When will v1 be deprecated?",
]

for question in TEST_QUESTIONS:
    result = rag.answer(question)
    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']} (top score: {result['top_score']:.3f})")
```

---

## Exercise 3 — Local vs API quality comparison (Stretch)

Run the same evaluation set through a local model (Ollama) and an API model (GPT-4o-mini). Score both using an LLM judge and report the quality gap.

```python
import os
from openai import OpenAI

local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
judge_client = api_client  # Always use API for judging

EVAL_SET = [
    {
        "question": "Explain why Transformers outperform RNNs for long sequences.",
        "ground_truth": "Transformers use self-attention with O(1) path length between any two tokens, avoiding the vanishing gradient problem of RNNs. RNNs process sequentially and can't parallelize, while Transformers process all positions simultaneously."
    },
    {
        "question": "What is the difference between precision and recall in information retrieval?",
        "ground_truth": "Precision = relevant retrieved / total retrieved. Recall = relevant retrieved / total relevant. High precision means most retrieved docs are relevant. High recall means most relevant docs were retrieved."
    },
    {
        "question": "Write a Python one-liner to flatten a list of lists.",
        "ground_truth": "[item for sublist in nested_list for item in sublist]"
    },
]

def get_answer(client: OpenAI, model: str, question: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=200
    )
    return resp.choices[0].message.content

def judge_answer(question: str, answer: str, ground_truth: str) -> float:
    resp = judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Rate this answer compared to the ground truth. Score 1-5.
Question: {question}
Ground truth: {ground_truth}
Answer: {answer}
Return JSON: {{"score": int, "reasoning": "one sentence"}}"""
        }],
        temperature=0.0,
        max_tokens=100,
        response_format={"type": "json_object"}
    )
    import json
    result = json.loads(resp.choices[0].message.content)
    return result["score"], result["reasoning"]

import statistics

local_scores = []
api_scores = []

print(f"{'Question':<50} {'Local':>6} {'API':>6}")
print("-"*65)

for item in EVAL_SET:
    local_ans = get_answer(local_client, "llama3.1:8b", item["question"])
    api_ans = get_answer(api_client, "gpt-4o-mini", item["question"])

    local_score, _ = judge_answer(item["question"], local_ans, item["ground_truth"])
    api_score, _ = judge_answer(item["question"], api_ans, item["ground_truth"])

    local_scores.append(local_score)
    api_scores.append(api_score)
    print(f"{item['question'][:50]:<50} {local_score:>6} {api_score:>6}")

print("-"*65)
print(f"{'Average':<50} {statistics.mean(local_scores):>6.2f} {statistics.mean(api_scores):>6.2f}")
gap = statistics.mean(api_scores) - statistics.mean(local_scores)
print(f"\nQuality gap: {gap:.2f} points (API is {gap/5*100:.0f}% better on 1-5 scale)")
```

**Expected:** API (GPT-4o-mini) scores 4.0–4.5/5 on average. Local Llama 3.1 8B scores 3.5–4.0/5. The gap narrows on factual questions and widens on complex reasoning. Use this measurement to decide whether the quality gap justifies the API cost for your specific use case.

---

[[05-when-to-run-locally]] | [[07-interview-questions]]
