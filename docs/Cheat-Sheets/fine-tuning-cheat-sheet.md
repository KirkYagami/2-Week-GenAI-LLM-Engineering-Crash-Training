# Fine-Tuning Cheat Sheet

Quick reference for QLoRA fine-tuning with PEFT and SFTTrainer.

---

## Setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"  # or "meta-llama/Llama-3.2-1B-Instruct"
```

## 4-bit quantization

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 > fp4 for most models
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Extra 0.1-0.4 bits per param savings
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
```

## LoRA config

```python
lora_config = LoraConfig(
    r=16,                          # Rank. 8–64. Higher = more params, more capacity.
    lora_alpha=32,                 # Scaling. Usually 2*r.
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Typical: < 1% of total params
```

## Training arguments

```python
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # Effective batch = 4*2 = 8
    learning_rate=2e-4,
    bf16=True,                        # Use bf16 if GPU supports it
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    optim="paged_adamw_32bit",        # Memory-efficient optimizer
    warmup_ratio=0.03,
    report_to="none",
)
```

## SFTTrainer

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    dataset_text_field="text",    # Column with formatted chat text
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
```

## Data format (chat template)

```python
# Each example: messages in OpenAI chat format
example = {
    "messages": [
        {"role": "system", "content": "Classify sentiment."},
        {"role": "user", "content": "I love this product!"},
        {"role": "assistant", "content": "positive"},
    ]
}

# Format with tokenizer's chat template
text = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False,  # False for training, True for inference
)
```

## Merge and save

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "./output")
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")
```

## Memory requirements (rough guide)

| Model | Full FT | QLoRA 4-bit | GPU |
|-------|---------|-------------|-----|
| 0.5B | 4GB | 2GB | Any modern |
| 1B | 8GB | 3GB | T4 |
| 7B | 28GB | 6GB | A100/T4 |
| 13B | 56GB | 10GB | A100 |

## When to fine-tune vs. prompt

| Situation | Choice |
|-----------|--------|
| < 50 examples per class | Prompt |
| Specific output format needed consistently | Fine-tune |
| Domain jargon not in base model | Fine-tune |
| Task changes monthly | Prompt |
| Need < 100ms inference at scale | Fine-tune (small model) |
