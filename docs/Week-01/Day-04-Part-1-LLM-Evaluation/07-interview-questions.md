# Interview Questions — LLM Evaluation

---

## Q1: What is RAGAS and which of its metrics require ground truth labels?

??? "Show answer"
    RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG pipelines. It provides four core metrics:

    **No ground truth required:**
    - **Faithfulness** — uses the LLM to check whether each claim in the answer is entailed by the retrieved context. No reference answer needed.
    - **Answer Relevancy** — generates back-questions from the answer and measures similarity to the original question via embeddings. No reference answer needed.

    **Ground truth required:**
    - **Context Precision** — measures what fraction of the retrieved chunks are relevant to answering the question. Needs a reference answer to judge relevance.
    - **Context Recall** — measures what fraction of the ground truth answer is covered by retrieved context. Needs the reference answer to decompose into claims.

    The practical implication: you can run faithfulness and answer relevancy at scale without annotation effort. Context precision and recall require a labeled eval set.

---

## Q2: What is the difference between faithfulness and factual accuracy?

??? "Show answer"
    These are often confused but measure different things:

    **Faithfulness** measures whether the answer adheres to the provided context. A faithful answer contains only information that can be found in the retrieved documents.

    **Factual accuracy** measures whether the answer is actually true in the real world.

    A faithful answer can be factually wrong — if the retrieved context contains incorrect information, a faithful answer will faithfully reproduce that incorrect information.

    Conversely, an unfaithful answer can be factually correct — the model might hallucinate a true fact from its parametric memory (e.g., adding "Python was released in 1991" when the context doesn't mention it).

    In RAG systems, you want both: faithfulness ensures the model doesn't add unsupported claims; factual accuracy requires the underlying documents to be correct. You control faithfulness with prompt engineering and grounding instructions; factual accuracy depends on your data quality.

---

## Q3: Explain the difference between context precision and context recall. What causes each to be low?

??? "Show answer"
    **Context precision** = of the chunks you retrieved, how many are actually relevant?
    - Low precision: you're retrieving noise — irrelevant chunks that dilute the good content
    - Root cause: poor embedding model, too-large k, chunking that mixes unrelated topics
    - Fix: better embedding model, reduce k, add reranking, improve chunking

    **Context recall** = of the ground truth answer's content, how much is covered by the retrieved chunks?
    - Low recall: you're missing the documents that contain the answer
    - Root cause: relevant documents not retrieved — poor embedding model, poor chunking, documents not in index
    - Fix: better chunking, add query expansion, hybrid search, check your ingestion pipeline

    Think of it with a precision/recall analogy from classification:
    - Precision: what fraction of what you returned was correct?
    - Recall: what fraction of what was correct did you return?

    A system with high precision, low recall returns few but relevant chunks — it's conservative.
    A system with high recall, low precision returns many chunks including relevant ones — it's noisy.

---

## Q4: What biases affect LLM-as-judge evaluation, and how do you mitigate them?

??? "Show answer"
    Four well-documented biases:

    **Verbosity bias:** Longer, more detailed answers score higher regardless of quality. Mitigation: evaluate each dimension separately (correctness vs. completeness vs. clarity); never ask for a single "quality" score.

    **Position/order bias:** When comparing two answers (A vs B), the first tends to win. Mitigation: run each comparison twice with A and B swapped; take the average or flag disagreements.

    **Self-preference:** GPT-4o rates GPT-4o outputs higher; Claude rates Claude outputs higher. Mitigation: use a different model as judge than the one being evaluated; use multiple judges.

    **Anchoring:** Seeing a bad example first makes subsequent examples seem better in comparison. Mitigation: randomize eval set order; use absolute rubrics rather than relative comparisons when anchoring is a concern.

    For production systems, validate your LLM judge on a human-labeled subset. If the judge's rankings correlate with human rankings (Spearman ρ > 0.7), it's trustworthy at scale.

---

## Q5: What is Cohen's kappa and when would you use it in an LLM evaluation context?

??? "Show answer"
    Cohen's kappa is a measure of inter-rater agreement that corrects for chance:

    `κ = (P_observed - P_expected) / (1 - P_expected)`

    Where:
    - P_observed = fraction of items where both raters agreed
    - P_expected = agreement expected by random chance, given the distribution of ratings

    **Why correct for chance?** If one rater gives "good" 80% of the time and another gives "good" 80% of the time, they'll agree ~64% of the time purely by chance even if their judgments are uncorrelated.

    **Interpretation:**
    - κ < 0.40: Poor — don't use these raters together; run calibration
    - 0.40–0.60: Moderate — acceptable for exploratory work
    - 0.60–0.80: Substantial — suitable for production annotation
    - κ > 0.80: Near-perfect — well-calibrated team

    **When to use:** Before using human annotators for production eval, measure kappa on a calibration set of ~30 examples. If kappa < 0.60, run a calibration session where raters discuss their disagreements before continuing.

    You can also measure kappa between an LLM judge and human raters to validate whether the LLM judge is trustworthy.

---

## Q6: How would you design an evaluation pipeline for a new RAG feature before shipping it?

??? "Show answer"
    A practical pre-ship eval pipeline has three stages:

    **Stage 1 — Automated regression suite (runs in CI)**
    - 50–100 labeled examples covering the feature's scope
    - RAGAS faithfulness and answer relevancy scored automatically
    - Retrieval Recall@5 on labeled relevant documents
    - Must pass: faithfulness > 0.80, answer relevancy > 0.75, Recall@5 > 0.85
    - Runtime: < 5 minutes

    **Stage 2 — LLM-as-judge on broader eval set (runs pre-merge)**
    - 200–500 examples including edge cases and adversarial inputs
    - GPT-4o judges correctness, completeness, and citation quality (1–5 rubric)
    - Compares scores vs the previous version (regression detection)
    - Must pass: no statistically significant regression on any dimension

    **Stage 3 — Human spot check (runs pre-launch)**
    - 20–30 examples selected from the hardest cases and new capability demonstrations
    - 2 internal raters score each example on the same rubric as Stage 2
    - Must pass: average human score ≥ 4.0 on correctness, kappa > 0.60
    - This is the gate that the automated pipeline can't replace

    Output: a one-page eval report summarizing all three stages, linked in the PR.

---

## Q7: What is Recall@k and MRR, and how do they differ?

??? "Show answer"
    Both evaluate retrieval quality given labeled relevant documents.

    **Recall@k** = what fraction of all relevant documents appear in the top k retrieved results?
    - Answers: "Does the retriever find the relevant documents?"
    - Example: if there are 3 relevant docs and 2 appear in the top 5, Recall@5 = 0.67
    - Best for: systems where you need all relevant documents (comprehensive recall matters)

    **MRR (Mean Reciprocal Rank)** = average of 1/rank_of_first_relevant_doc across queries
    - Answers: "How high does the first relevant document appear?"
    - Example: if the first relevant doc is at rank 2, RR = 0.5. Averaged across queries = MRR
    - Best for: systems where users look at the top result and move on (position matters most)

    Key difference: Recall@k cares about finding all relevant docs; MRR cares about finding at least one relevant doc as high as possible.

    For RAG: Recall@k is usually more important because missing a relevant document means the LLM won't have access to it. A high MRR with low recall means the best answer ranks first but other important context is missing.

---

[[06-practice-exercises]] | [[../Day-04-Part-2-Responsible-AI-and-Safety/00-agenda]]
