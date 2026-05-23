# Advanced Features — Fine-Tuned Classifier

## Label masking: only train on the response

By default, the loss is computed over all tokens. For classification, you want the model to learn to predict the label — not to reproduce the system prompt or user message. Label masking sets the loss to -100 for non-assistant tokens.

SFTTrainer handles this when you use the `DataCollatorForCompletionOnlyLM`:

```python
from trl import DataCollatorForCompletionOnlyLM

# Tell the collator which token(s) mark the start of the assistant's response
response_template = "<|im_start|>assistant\n"
response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_token_ids,
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    ...
    data_collator=collator,
)
```

With label masking, the training loss only reflects the model's ability to predict "positive", "negative", or "neutral" — not to reconstruct the prompt. This significantly improves classification accuracy.

## Tracking training metrics

Enable logging to see loss per step:

```python
training_args = TrainingArguments(
    ...
    logging_steps=5,
    logging_dir="./logs",
    report_to="none",  # Or "wandb" if you have W&B set up
)
```

Expected training loss curve:
- Step 0: loss ~1.5 (random guessing across 3 classes)
- Step 50: loss ~0.3 (learning the task)
- Step 100+: loss ~0.1 (converging)

If loss doesn't decrease past ~0.5, check: learning rate (try 1e-4 or 3e-4), batch size, dataset quality (are labels consistent?).

## Inference optimization: GGUF quantization

For CPU deployment, convert the merged model to GGUF format using llama.cpp:

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make

# Convert to GGUF Q4_K_M (good quality/size tradeoff)
python convert_hf_to_gguf.py ../merged-model --outtype q4_k_m --outfile ../classifier-q4.gguf
```

This reduces the 0.5B model from ~1GB (float16) to ~350MB (Q4_K_M) and runs fast on CPU.

## Comparing few-shot prompting vs. fine-tuning

For a complete portfolio comparison, also evaluate 5-shot prompting:

```python
FEW_SHOT_EXAMPLES = """
Review: "Absolutely loved this product! Works perfectly."
Sentiment: positive

Review: "Mediocre at best. Nothing special."
Sentiment: neutral

Review: "Broke after one week. Terrible quality."
Sentiment: negative

Review: "Great value for money, highly recommend."
Sentiment: positive

Review: "Does what it says, nothing more."
Sentiment: neutral
"""

async def predict_few_shot(text: str) -> str:
    from openai import AsyncOpenAI
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify review sentiment. Respond with one word: positive, negative, or neutral."},
            {"role": "user", "content": f"{FEW_SHOT_EXAMPLES}\nReview: \"{text}\"\nSentiment:"},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().lower()
```

---

[[02-implementation]] | [[04-evaluation]]
