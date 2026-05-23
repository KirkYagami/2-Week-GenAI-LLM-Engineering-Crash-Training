# Implementation — Fine-Tuned Classifier

## QLoRA fine-tuning

```python
# train.py
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./qlora-checkpoint"

# ---- 4-bit quantization config ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---- Load model and tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ---- LoRA config ----
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~2M / 500M total = ~0.4%

# ---- Load dataset ----
def format_for_training(example: dict) -> dict:
    """Apply chat template and return formatted text."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})
dataset = dataset.map(format_for_training)

# ---- Training arguments ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    optim="paged_adamw_32bit",
    warmup_ratio=0.03,
    report_to="none",
)

# ---- Trainer ----
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Training complete. Checkpoint saved to {OUTPUT_DIR}")
```

## Merge and save

```python
# merge.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
CHECKPOINT = "./qlora-checkpoint"
MERGED_PATH = "./merged-model"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, CHECKPOINT)

print("Merging weights...")
merged = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
merged.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("Done.")
```

## FastAPI inference endpoint

```python
# app.py
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

SYSTEM_PROMPT = (
    "Classify the sentiment of the following product review. "
    "Respond with exactly one word: positive, negative, or neutral."
)
VALID_LABELS = {"positive", "negative", "neutral"}

model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model_path = os.getenv("MODEL_PATH", "./merged-model")
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    print("Model loaded.")
    yield

app = FastAPI(title="Sentiment Classifier", lifespan=lifespan)

class ClassifyRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify(request: ClassifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    label = next((l for l in VALID_LABELS if l in raw), "unknown")
    return {"label": label, "raw_output": raw}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
```

> [!tip] Use `max_new_tokens=5` for classification
> Classification should output one word. Setting `max_new_tokens=5` prevents the model from generating lengthy explanations and speeds up inference significantly.

---

[[01-setup]] | [[03-advanced-features]]
