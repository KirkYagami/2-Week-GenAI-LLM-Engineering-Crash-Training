# Chain-of-Thought

CoT is one of the most asked-about prompting techniques in interviews — not because it's complicated, but because understanding *why* it works reveals how LLMs actually process information.

---

## Q1: What is chain-of-thought prompting and when does it help?

??? "Show answer"
    Chain-of-thought (CoT) prompting encourages the model to generate intermediate reasoning steps before the final answer. Instead of jumping to the answer, the model "thinks out loud."

    It helps when:
    - The task requires multiple reasoning steps (math, logic, multi-hop Q&A)
    - Errors in intermediate steps are the primary failure mode
    - The answer space is large and the model needs to narrow it systematically

    It doesn't help when:
    - The task is simple classification or direct retrieval — extra tokens waste money
    - The model is small (< 7B) — CoT requires sufficient capacity to reason coherently
    - You need strict structured output — reasoning prose before JSON can break parsers

    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"{question}\n\nLet's think step by step."
        }]
    )
    ```

---

## Q2: What's the difference between zero-shot CoT and few-shot CoT?

??? "Show answer"
    **Zero-shot CoT**: Append "Let's think step by step" to the prompt. The model generates its own reasoning chain. Simple to implement, works well on capable models.

    **Few-shot CoT**: Include 3–8 examples where each shows the full reasoning chain before the answer. The model learns the *style* of reasoning you want, not just that it should reason.

    ```python
    FEW_SHOT_COT = """
    Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each. How many does he have?
    A: Roger starts with 5 balls. 2 cans × 3 balls = 6 new balls. 5 + 6 = 11. The answer is 11.

    Q: There are 15 trees. After today, 21 trees will be planted. How many total?
    A: Start: 15. Planted: 21. Total: 15 + 21 = 36. The answer is 36.

    Q: {question}
    A:"""
    ```

    Few-shot CoT generally outperforms zero-shot on complex reasoning, but requires maintaining and validating examples. Errors in your examples compound into the model's output.

---

## Q3: What is self-consistency and how does it improve CoT?

??? "Show answer"
    Self-consistency samples the model multiple times with temperature > 0, generates multiple independent reasoning chains, then takes the **majority vote** on the final answer.

    If 8 out of 10 independent reasoning paths arrive at the same answer, that answer is more reliable than any single path.

    ```python
    from collections import Counter
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def self_consistent_answer(question: str, samples: int = 10) -> str:
        answers = []
        for _ in range(samples):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"{question}\n\nLet's think step by step."}],
                temperature=0.7,
            )
            text = response.choices[0].message.content
            answers.append(text.split("\n")[-1].strip())

        most_common, _ = Counter(answers).most_common(1)[0]
        return most_common
    ```

    Cost: 10× the tokens. Use only for high-stakes tasks where accuracy justifies the cost.

---

## Q4: When does chain-of-thought hurt performance?

??? "Show answer"
    1. **Simple tasks** — reasoning about "what is 2+2" can confuse the model into second-guessing a correct answer
    2. **Small models** — under ~7B parameters, models often produce incoherent or circular chains that lead to wrong answers
    3. **Wrong reasoning chain** — a confident but incorrect chain leads to a confidently wrong answer. CoT amplifies the model's reasoning, both correct and incorrect
    4. **Strict output format** — reasoning prose before JSON breaks parsers. Either extract the answer from the chain, or use function calling instead
    5. **Latency-critical applications** — CoT generates more tokens, increasing time-to-first-token and total cost

---

## Q5: How do you use CoT for structured output without breaking the JSON parser?

??? "Show answer"
    Two approaches:

    **Option 1 — Reason then output in delimited blocks**:

    ```python
    PROMPT = """Analyze the support ticket, then output JSON.

    First think through: What is the customer asking for? What is the severity? What category?

    Then output your answer between <json> and </json> tags.

    Ticket: {ticket}"""

    import re

    def extract_json(response: str) -> str:
        match = re.search(r"<json>(.*?)</json>", response, re.DOTALL)
        return match.group(1).strip() if match else response
    ```

    **Option 2 — Two-call chain**: First call reasons in prose, second call converts reasoning to JSON only.

    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    reasoning = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Analyze this ticket: {ticket}. Think through the category, severity, and next steps."}],
    ).choices[0].message.content

    structured = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Based on this analysis:\n{reasoning}\n\nOutput ONLY valid JSON with keys: category, severity, summary."}],
        response_format={"type": "json_object"},
    ).choices[0].message.content
    ```

    Option 2 is more reliable for complex schemas but doubles API cost and latency.

---

*Previous: [Prompt Design](01-prompt-design.md) | Next: [Structured Output](03-structured-output.md)*
