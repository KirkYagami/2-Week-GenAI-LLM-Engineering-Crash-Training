# Training Data

Training data is the highest-leverage variable in fine-tuning. A well-trained model on 500 high-quality examples will outperform one trained on 50,000 mediocre examples. Interviewers ask about data to see if you've thought about quality, not just quantity.

---

## Q1: What format does training data need to be in for instruction fine-tuning?

??? "Show answer"
    Modern instruction fine-tuning uses the **chat template format** — the same format used at inference time. Each example is a list of messages that the tokenizer converts using `apply_chat_template`.

    ```python
    # train.jsonl — one JSON object per line
    {"messages": [
        {"role": "system", "content": "You are a helpful customer support agent for Acme Corp."},
        {"role": "user", "content": "I was charged twice last month. What should I do?"},
        {"role": "assistant", "content": "I'm sorry to hear that. Please share your order number and I'll check it immediately. Duplicate charges are typically resolved within 3-5 business days."}
    ]}
    ```

    Format the dataset before training:

    ```python
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    def format_example(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    formatted_dataset = dataset.map(format_example)
    ```

    The formatted `text` field is what `SFTTrainer` with `dataset_text_field="text"` expects.

---

## Q2: How much data do you need for LoRA fine-tuning?

??? "Show answer"
    General guidelines:

    | Task | Minimum examples | Notes |
    |------|-----------------|-------|
    | Style/tone adaptation | 50–200 | The base model already knows the task |
    | Domain-specific format | 200–500 | Teaching a new output schema |
    | New knowledge / domain | 500–5,000 | Facts must appear in training data |
    | Classification | 100–1,000 | Depends on number of classes and ambiguity |

    **Quality beats quantity.** 200 carefully written examples that clearly demonstrate the target behaviour outperform 2,000 examples scraped from the web.

    Signs you need more data:
    - Validation loss plateaus early and the model still makes systematic errors
    - High variance in model outputs on similar inputs

    Signs you have a data quality problem (more data won't help):
    - Training loss decreases but validation loss increases (overfitting)
    - Model performs well on training examples but fails on held-out test cases
    - Model copies training examples verbatim rather than generalizing

---

## Q3: What makes a good training example vs a bad one?

??? "Show answer"
    **Good examples**:
    - Clearly demonstrate the target behaviour with no ambiguity
    - Cover edge cases the model will encounter in production
    - Are diverse — different phrasing, different topics, different lengths
    - The assistant response is exactly what you want the model to do

    **Bad examples**:
    - Contain errors or inconsistencies in the assistant response
    - Are near-duplicates of each other (model memorizes rather than generalizes)
    - Include behaviours you don't want the model to learn
    - Are too short or vague to demonstrate the pattern

    Data quality checklist:
    ```python
    def validate_example(example: dict) -> list[str]:
        issues = []
        messages = example.get("messages", [])

        if not messages:
            issues.append("Empty messages")
        if messages[-1]["role"] != "assistant":
            issues.append("Last message must be from assistant")
        if len(messages[-1]["content"]) < 20:
            issues.append("Assistant response too short")

        # Check for near-duplicates by hashing assistant responses
        response_hash = hash(messages[-1]["content"])
        # Track and flag duplicates in your pipeline

        return issues
    ```

---

## Q4: How do you prevent your fine-tuned model from forgetting general capabilities?

??? "Show answer"
    Four strategies:

    1. **Data mixing** — include 10–20% general-purpose examples in your training dataset alongside task-specific data. This is the most effective approach.

    ```python
    # Mix 80% task data + 20% general data
    task_data = load_dataset("json", data_files="task_examples.jsonl")
    general_data = load_dataset("Open-Orca/OpenOrca", split="train").shuffle(seed=42).select(range(1000))
    mixed = concatenate_datasets([task_data, general_data]).shuffle(seed=42)
    ```

    2. **Low learning rate** — use `2e-4` or lower to avoid overwriting base knowledge with large gradient updates.

    3. **Short training** — 1–3 epochs is usually enough. Longer training increases the risk of forgetting.

    4. **Evaluate on a general benchmark** — run the model on MMLU or HellaSwag before and after fine-tuning. If general benchmark scores drop significantly, reduce training time or add more general data to the mix.

---

## Q5: What is data contamination and how do you detect it?

??? "Show answer"
    **Data contamination** occurs when your evaluation set overlaps with your training set — your model scores well on evals because it has already "seen" the answers, not because it has generalized.

    Detection:

    ```python
    from difflib import SequenceMatcher

    def find_contaminated(train_data: list[str], eval_data: list[str], threshold: float = 0.85) -> list[int]:
        contaminated = []
        for i, eval_example in enumerate(eval_data):
            for train_example in train_data:
                similarity = SequenceMatcher(None, eval_example, train_example).ratio()
                if similarity > threshold:
                    contaminated.append(i)
                    break
        return contaminated
    ```

    For large datasets, use MinHash or SimHash for approximate deduplication — exact comparison is O(n²).

    Prevention:
    - Split your data before any preprocessing or augmentation
    - Generate your eval set independently from a different source than your training data
    - If using a public benchmark, verify your training data doesn't include that benchmark's test set

---

*Previous: [LoRA and QLoRA](01-lora-qlora.md) | Next: [Evaluation](03-evaluation.md)*
