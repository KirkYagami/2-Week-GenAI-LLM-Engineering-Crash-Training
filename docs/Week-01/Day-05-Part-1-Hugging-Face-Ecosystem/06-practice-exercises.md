# Practice Exercises — Hugging Face Ecosystem

---

## Exercise 1 — Multi-task inference pipeline (Warm-up)

Use the Inference API to run three different NLP tasks on the same user review, and combine the results into a structured report.

```python
import os
import json
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

REVIEW = """
I've been using this app for 3 months. The interface is clean and easy to navigate.
However, I've had two occasions where the app crashed and I lost my work.
The customer support team was responsive and helpful when I reported the issues.
Overall, it's a good product that just needs some stability improvements.
"""

def analyze_review(review: str) -> dict:
    results = {}

    # Task 1: Sentiment
    sentiment = client.text_classification(
        text=review,
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    results["sentiment"] = {
        "label": sentiment[0]["label"],
        "confidence": round(sentiment[0]["score"], 3)
    }

    # Task 2: Department classification
    topics = client.zero_shot_classification(
        text=review,
        labels=["product quality", "customer support", "stability/bugs", "user interface"],
        multi_label=True,
        model="facebook/bart-large-mnli"
    )
    results["topics"] = {
        label: round(score, 3)
        for label, score in zip(topics["labels"], topics["scores"])
        if score > 0.4
    }

    # Task 3: Summarize (NER to extract key entities)
    entities = client.token_classification(
        text=review,
        model="dslim/bert-base-NER"
    )
    results["entities"] = [
        {"text": e["word"], "type": e["entity_group"]}
        for e in entities
        if e["entity_group"] in ("ORG", "PRODUCT") and e["score"] > 0.8
    ]

    return results

report = analyze_review(REVIEW)
print(json.dumps(report, indent=2))
# Expected output:
# {
#   "sentiment": {"label": "POSITIVE", "confidence": 0.782},
#   "topics": {"stability/bugs": 0.95, "customer support": 0.88, "user interface": 0.72},
#   "entities": []
# }
```

---

## Exercise 2 — Build and deploy a RAG evaluation Gradio app (Main)

Build a Gradio app that accepts a question, a context, and a generated answer, and scores them using a faithfulness heuristic. Deploy it to a Space (or run locally).

```python
# app.py
import os
import json
import gradio as gr
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def score_faithfulness(question: str, context: str, answer: str) -> tuple[str, str]:
    """Score answer faithfulness against context using GPT-4o-mini."""
    if not question.strip() or not context.strip() or not answer.strip():
        return "⚠ Please fill in all three fields.", ""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Evaluate if this answer is faithful to the context.

Context: {context}
Question: {question}
Answer: {answer}

Check each claim in the answer against the context.
Return JSON:
{{
  "score": 0.0-1.0,
  "faithful": true/false,
  "claims": [
    {{"text": "claim text", "supported": true/false, "reason": "brief reason"}}
  ],
  "verdict": "one sentence summary"
}}"""
        }],
        temperature=0.0,
        max_tokens=400,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    score = result["score"]
    
    # Build score display
    score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
    status = "✓ FAITHFUL" if result["faithful"] else "✗ HALLUCINATION DETECTED"
    
    summary = f"""**Score: {score:.2f}/1.00** [{score_bar}]  {status}

**Verdict:** {result['verdict']}
"""
    
    # Build claims breakdown
    claims_md = "**Claim-by-claim breakdown:**\n\n"
    for claim in result.get("claims", []):
        icon = "✓" if claim["supported"] else "✗"
        claims_md += f"{icon} _{claim['text']}_\n   → {claim['reason']}\n\n"

    return summary, claims_md

demo = gr.Blocks(theme=gr.themes.Soft(), title="RAG Faithfulness Checker")

with demo:
    gr.Markdown("# RAG Faithfulness Checker\nScore whether a generated answer is faithful to the retrieved context.")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(label="Question", placeholder="What question was asked?", lines=2)
            context = gr.Textbox(label="Retrieved context", placeholder="Paste the context that was given to the LLM...", lines=6)
            answer = gr.Textbox(label="Generated answer", placeholder="Paste the LLM's answer...", lines=4)
            btn = gr.Button("Score faithfulness", variant="primary")

        with gr.Column():
            score_output = gr.Markdown(label="Score")
            claims_output = gr.Markdown(label="Claims breakdown")

    btn.click(
        fn=score_faithfulness,
        inputs=[question, context, answer],
        outputs=[score_output, claims_output]
    )

    gr.Examples(
        examples=[
            [
                "When was Python released?",
                "Python is a programming language created by Guido van Rossum. It was first released in 1991.",
                "Python was released in 1991 by Guido van Rossum."
            ],
            [
                "What is RAG?",
                "RAG stands for Retrieval-Augmented Generation.",
                "RAG stands for Retrieval-Augmented Generation. It was invented at Stanford in 2019."
            ],
        ],
        inputs=[question, context, answer]
    )

if __name__ == "__main__":
    demo.launch()
```

**Deploying to a Space:**

```bash
# 1. Create requirements.txt
echo "gradio>=4.0
openai>=1.0" > requirements.txt

# 2. Push to a Space
huggingface-cli repo create rag-faithfulness-checker --type space --space_sdk gradio
git clone https://huggingface.co/spaces/your-username/rag-faithfulness-checker
cp app.py requirements.txt rag-faithfulness-checker/
cd rag-faithfulness-checker && git add . && git commit -m "add faithfulness checker" && git push

# 3. Add secret via SDK
python -c "
from huggingface_hub import add_space_secret
import os
add_space_secret('your-username/rag-faithfulness-checker', 'OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
"
```

---

## Exercise 3 — Benchmark embedding models (Stretch)

Compare three embedding models from the Hub on the MTEB STS (semantic textual similarity) benchmark. Report the correlation between model similarity scores and human similarity ratings.

```python
import os
import numpy as np
from scipy.stats import spearmanr
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

# Mini STS benchmark — (sentence1, sentence2, human_similarity_1-5)
STS_PAIRS = [
    ("A dog is running in the park.", "A canine is sprinting through a garden.", 4.5),
    ("The cat sat on the mat.", "The feline rested on the rug.", 4.0),
    ("She loves pizza.", "He hates ice cream.", 1.0),
    ("The economy grew by 3%.", "GDP increased 3 percent.", 4.8),
    ("How do I reset my password?", "I forgot my login credentials.", 3.5),
    ("The movie was excellent.", "The film was outstanding.", 4.7),
    ("I need to cancel my subscription.", "The weather is nice today.", 0.5),
    ("Machine learning models improve with data.", "More data helps train better ML systems.", 4.2),
]

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
]

def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def evaluate_model(model_id: str, pairs: list[tuple]) -> dict:
    """Compute Spearman correlation between model similarities and human ratings."""
    model_sims = []
    human_sims = []

    for sent1, sent2, human_score in pairs:
        embs = client.feature_extraction(
            text=[sent1, sent2],
            model=model_id,
            normalize=True
        )
        sim = cosine_sim(embs[0], embs[1])
        model_sims.append(sim)
        human_sims.append(human_score)

    rho, p_value = spearmanr(model_sims, human_sims)
    return {
        "model": model_id,
        "spearman_rho": round(rho, 4),
        "p_value": round(p_value, 4),
        "avg_similarity": round(np.mean(model_sims), 4)
    }

print(f"{'Model':<50} {'Spearman ρ':>12} {'p-value':>10}")
print("-" * 75)
for model_id in MODELS:
    result = evaluate_model(model_id, STS_PAIRS)
    print(f"{result['model']:<50} {result['spearman_rho']:>12.4f} {result['p_value']:>10.4f}")
```

**Expected pattern:** `bge-small-en-v1.5` typically achieves the highest Spearman ρ on English STS. Lower-parameter models trade accuracy for speed. Use this pattern to select the best model for your specific domain.

---

[[05-datasets-and-models]] | [[07-interview-questions]]
