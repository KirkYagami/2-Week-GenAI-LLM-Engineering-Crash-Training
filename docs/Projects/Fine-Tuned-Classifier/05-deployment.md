# Deployment — Fine-Tuned Classifier

## Docker for the inference server

```dockerfile
FROM python:3.11-slim
WORKDIR /app

# CPU-only torch (smaller image)
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY app.py .
COPY merged-model/ ./merged-model/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```
# requirements-inference.txt (CPU-only, no training deps)
transformers==4.45.0
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
python-dotenv==1.0.1
```

> [!info] The inference container doesn't need GPU
> After merging, the weights are standard PyTorch tensors. CPU inference is viable for a 0.5B model: ~200–400ms per classification, vs. < 50ms on GPU. For batch processing, CPU is often sufficient. For < 100ms latency requirements, use a GPU instance.

## Fly.io deployment

```toml
# fly.toml
app = 'sentiment-classifier'
primary_region = 'iad'

[build]
  dockerfile = 'Dockerfile'

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[[vm]]
  memory = "2gb"    # Model weights need ~1GB RAM
  cpu_kind = "shared"
  cpus = 2
```

```bash
flyctl launch --name sentiment-classifier
flyctl deploy
```

> [!warning] Cold start for local models is 10–30 seconds
> Loading a 0.5B model from disk takes 10–30 seconds on the first request. Set `min_machines_running = 1` to keep an instance warm, or accept the cold start and show a loading indicator in your UI.

## Pushing the model to HuggingFace Hub

Share your fine-tuned model and demonstrate open-source contributions:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./merged-model",
    repo_id="your-username/qwen2-0.5b-sentiment",
    repo_type="model",
)
```

Then in `app.py`, replace `MODEL_PATH` with the Hub repo ID — Transformers will download it automatically:

```python
MODEL_ID = "your-username/qwen2-0.5b-sentiment"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)
```

---

[[04-evaluation]] | [[06-interview-questions]]
