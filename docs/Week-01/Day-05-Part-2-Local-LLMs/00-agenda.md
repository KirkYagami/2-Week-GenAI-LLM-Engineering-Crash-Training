# Agenda — Local LLMs

Running an LLM locally means no API costs, no data leaving your machine, and no rate limits. The tradeoff is hardware requirements and engineering effort. This session teaches you to make the right call — and to execute it when local is the right answer.

## Learning objectives

By the end of this session you will be able to:

- Install and use Ollama for local model serving
- Understand what quantization does to model quality and memory
- Use `llama.cpp` for CPU-only inference
- Make a data-driven decision about local vs API inference

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:20 | Why run locally — privacy, cost, latency | [[01-why-run-locally]] |
| 0:20 – 1:00 | Ollama — install, run, API | [[02-ollama]] |
| 1:00 – 1:40 | llama.cpp — GGUF, CPU inference, Python bindings | [[03-llama-cpp]] |
| 1:40 – 2:10 | Quantization — formats, quality tradeoffs | [[04-quantization]] |
| 2:10 – 2:30 | Decision framework — when to run locally | [[05-when-to-run-locally]] |
| 2:30 – 3:00 | Practice exercises | [[06-practice-exercises]] |

## Setup

```bash
# Ollama (download from https://ollama.com)
# macOS:
brew install ollama

# Or download directly
curl -fsSL https://ollama.com/install.sh | sh

# llama.cpp Python bindings
pip install llama-cpp-python

# Verify Ollama
ollama pull llama3.2:3b   # 2 GB, runs on most machines
ollama run llama3.2:3b "Hello, what can you do?"
```

> [!info] Hardware requirements for this session
> Minimum: 8 GB RAM → run 3B or 7B models in Q4 quantization
> Comfortable: 16 GB RAM + any GPU → run 7B–13B models
> No GPU required for the Ollama and llama.cpp exercises — CPU inference works, just slower (~5–15 tokens/sec vs 30–100+ on GPU).

[[../Day-05-Part-1-Hugging-Face-Ecosystem/07-interview-questions|← Day 05 Part 1]] | [[01-why-run-locally|Start →]]
