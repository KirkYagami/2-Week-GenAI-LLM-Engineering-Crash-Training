# Vision and Multimodal

GPT-4o and Claude Sonnet 4.6 are natively multimodal — they accept images, PDFs, and documents alongside text. This unlocks a whole class of applications: receipt processing, chart reading, document extraction, visual QA.

## Learning objectives

- Pass images via URL and base64 encoding to both APIs
- Process PDFs through Anthropic's document API
- Extract structured data from images with vision + structured output
- Understand resolution tiers and their cost implications

---

## Sending images — OpenAI

Images are passed as content array items with type `image_url`.

```python
import os
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Method 1: Image URL (fastest — model downloads directly)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}
            },
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }],
    max_tokens=100
)
print(response.choices[0].message.content)

# Method 2: Base64 (required for local files or private images)
def encode_image(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(path).suffix.lstrip(".").lower().replace("jpg", "jpeg")
    return data, f"image/{ext}"

def describe_local_image(image_path: str, question: str) -> str:
    img_data, media_type = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{img_data}"}
                },
                {"type": "text", "text": question}
            ]
        }],
        max_tokens=300
    )
    return response.choices[0].message.content
```

---

## Image detail levels

OpenAI offers two resolution modes that affect token cost:

| Detail | Tile strategy | Tokens per image | Use when |
|--------|---------------|-----------------|----------|
| `low` | Single 512×512 downsample | ~85 tokens | Quick classification, general description |
| `high` | Multiple 512×512 tiles | 85 + 170 per tile | Text in images, detailed charts, documents |
| `auto` (default) | Model decides | Varies | Let OpenAI optimize |

```python
# Low detail — cheap, fast, good for general classification
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/product.jpg",
                    "detail": "low"
                }
            },
            {"type": "text", "text": "Is this a shirt or a jacket?"}
        ]
    }],
    max_tokens=20
)
```

> [!tip] Use `detail: "low"` for bulk processing
> Processing 10,000 product images at `detail: "low"` costs ~85 tokens vs ~765 tokens for a 1024×1024 image at high detail. That's a 9× cost reduction for tasks that don't need fine-grained detail.

---

## Sending images — Anthropic

Anthropic's image API uses a `source` dict rather than `image_url`.

```python
import os
import base64
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# URL source
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=300,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                }
            },
            {"type": "text", "text": "Describe this image briefly."}
        ]
    }]
)
print(response.content[0].text)

# Base64 source for local files
def describe_image_anthropic(image_path: str, question: str) -> str:
    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = image_path.rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{ext}",
                        "data": img_data
                    }
                },
                {"type": "text", "text": question}
            ]
        }]
    )
    return response.content[0].text
```

---

## PDF processing — Anthropic

Claude Sonnet 4.6 natively processes PDFs up to 100 pages without any preprocessing.

```python
import base64

def extract_from_pdf(pdf_path: str, question: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_data
                    }
                },
                {"type": "text", "text": question}
            ]
        }]
    )
    return response.content[0].text

# Extract payment terms from a contract
# answer = extract_from_pdf("contract.pdf", "List all payment terms mentioned.")
```

> [!info] OpenAI PDF support
> As of 2025, OpenAI supports PDFs via the Responses API (`input_file` parameter) and the Assistants API. For Chat Completions, convert PDF pages to images first using `pdf2image` or extract text with `pdfminer`.

---

## Structured extraction from images

Combine vision with structured output to extract typed data from documents.

```python
from pydantic import BaseModel
from typing import Optional

class Receipt(BaseModel):
    merchant_name: str
    date: str
    total_amount: float
    currency: str
    items: list[str]
    tax: Optional[float] = None

def extract_receipt(image_path: str) -> Receipt:
    img_data, media_type = encode_image(image_path)

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{img_data}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "Extract all receipt information from this image."
                }
            ]
        }],
        response_format=Receipt,
        max_tokens=500
    )
    return response.choices[0].message.parsed

# receipt = extract_receipt("receipt.jpg")
# print(f"Total: {receipt.total_amount} {receipt.currency}")
# print(f"Items: {receipt.items}")
```

---

## Multi-image requests

Both APIs support multiple images in a single message.

```python
def compare_images(image_paths: list[str], question: str) -> str:
    content = []
    for i, path in enumerate(image_paths):
        img_data, media_type = encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{img_data}"}
        })
        content.append({"type": "text", "text": f"Image {i + 1} above."})

    content.append({"type": "text", "text": question})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=500
    )
    return response.choices[0].message.content

# compare_images(["before.jpg", "after.jpg"], "What changed between these two images?")
```

---

## Common mistakes

> [!warning] Image size limits
> OpenAI: max 20MB per image, 2048×2048 px for high detail. Anthropic: max 5MB per image. Pre-resize large images before sending — a 24MP phone photo will often exceed these limits and cause a 400 error.

> [!warning] Base64 inflates payload size by ~33%
> A 3MB image becomes ~4MB base64 encoded. For bulk processing, use URL sources where possible — the model fetches directly and you avoid the overhead in your HTTP payload.

> [!warning] PDFs are not images
> Don't screenshot PDFs and send as images if you need text extraction — OCR accuracy degrades on compressed screenshots. Use Anthropic's native document API or extract text with `pdfminer` first.

---

[[01-openai-chat-completions]] | [[03-tool-use-openai]]
