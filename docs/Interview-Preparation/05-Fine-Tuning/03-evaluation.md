# Evaluation

Fine-tuning without evaluation is fine-tuning blind. These questions test whether you know how to measure what you actually changed — and whether you improved the right thing.

---

## Q1: How do you evaluate a fine-tuned model?

??? "Show answer"
    Evaluate at three levels:

    **1. Training metrics** (during training):
    - Training loss — should decrease steadily
    - Validation loss — should decrease with training loss; if it diverges, you're overfitting

    **2. Task-specific metrics** (offline):
    - For classification: accuracy, F1, precision, recall on a held-out test set
    - For generation: ROUGE, BERTScore, or LLM-as-judge on a set of test cases with expected outputs
    - For extraction: field-level F1 against ground truth

    **3. Comparative evaluation** (vs baseline):
    - Compare fine-tuned model against the prompt-only baseline on the same test set
    - Report the delta: "fine-tuned model improved F1 from 0.71 to 0.89 on 200 test cases"

    ```python
    from sklearn.metrics import classification_report

    y_true = ["billing", "technical", "account", "billing"]
    y_pred = [predict(text) for text in test_texts]

    print(classification_report(y_true, y_pred))
    # precision    recall  f1-score   support
    # billing       0.92      0.95      0.93      ...
    ```

---

## Q2: What metrics do you use to compare fine-tuned vs prompt-only baseline?

??? "Show answer"
    Choose metrics based on your task:

    | Task | Primary metric | Secondary metric |
    |------|----------------|-----------------|
    | Classification | F1 (macro) | Accuracy, per-class F1 |
    | Named entity extraction | Field-level F1 | Exact match rate |
    | Summarization | LLM-as-judge faithfulness | ROUGE-L |
    | Instruction following | LLM-as-judge (0-5 scale) | Format adherence rate |
    | Code generation | Pass@k (unit test pass rate) | Compilation rate |

    For format-sensitive tasks, always measure **format adherence** separately from content quality — a model that produces perfect JSON 95% of the time vs 60% is a different operational reality.

    ```python
    import json

    def format_adherence_rate(outputs: list[str]) -> float:
        valid = 0
        for output in outputs:
            try:
                json.loads(output)
                valid += 1
            except json.JSONDecodeError:
                pass
        return valid / len(outputs)

    baseline_rate = format_adherence_rate(baseline_outputs)   # e.g., 0.62
    finetuned_rate = format_adherence_rate(finetuned_outputs) # e.g., 0.97
    ```

---

## Q3: What is perplexity and when is it useful?

??? "Show answer"
    **Perplexity** measures how surprised a language model is by a held-out text. Lower perplexity = the model considers the text more probable = the model has learned the distribution better.

    `perplexity = exp(average negative log-likelihood)`

    ```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
    ```

    Perplexity is useful for:
    - Comparing model versions on the same domain text
    - Detecting catastrophic forgetting (perplexity on general text rises after fine-tuning)
    - Selecting the best checkpoint during training

    It does **not** directly measure task performance. A model with low perplexity on medical text can still answer medical questions incorrectly — perplexity measures fluency in the distribution, not factual accuracy.

---

## Q4: How do you detect overfitting during training?

??? "Show answer"
    **Overfitting** occurs when training loss decreases but validation loss increases — the model is memorizing training examples rather than generalizing.

    Detection:

    ```python
    from transformers import TrainingArguments, Trainer

    args = TrainingArguments(
        output_dir="./checkpoints",
        eval_strategy="steps",         # evaluate on validation set regularly
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,   # keep the checkpoint with best eval loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    ```

    Watch for:
    - `eval_loss` increasing while `train_loss` continues to decrease
    - Model generating training examples verbatim on held-out inputs
    - Large gap between training and eval loss early in training

    Mitigations:
    - Reduce number of training epochs (often 1–3 is enough for LoRA)
    - Increase `lora_dropout` (0.05–0.1)
    - Add more diverse training data
    - Increase `weight_decay` in optimizer

---

## Q5: What's different about evaluating generation tasks vs classification tasks?

??? "Show answer"
    **Classification** has a ground truth label — accuracy, F1, and confusion matrices apply directly. Evaluation is deterministic.

    **Generation** doesn't have a single correct answer — there are many valid responses to "summarize this document." Traditional metrics like ROUGE measure n-gram overlap but correlate poorly with human quality judgment.

    Better approaches for generation:

    **LLM-as-judge**: use a capable model to score outputs on a rubric.

    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def llm_judge(question: str, answer: str, context: str) -> dict:
        prompt = f"""Rate this answer on two dimensions (1-5 scale each):
    - Faithfulness: Does the answer only contain information from the context?
    - Relevance: Does the answer address the question?

    Context: {context}
    Question: {question}
    Answer: {answer}

    Reply as JSON: {{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one sentence>"}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        import json
        return json.loads(response.choices[0].message.content)
    ```

    Always run LLM-as-judge on a held-out test set (not training data) and report aggregate scores with standard deviation — individual scores are noisy.

---

*Previous: [Training Data](02-training-data.md) | Next: [Tracing and Monitoring](../06-LLMOps/01-tracing-and-monitoring.md)*
