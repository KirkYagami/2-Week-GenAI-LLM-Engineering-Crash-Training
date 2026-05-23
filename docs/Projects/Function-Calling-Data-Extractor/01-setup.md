# Setup and Environment — Function-Calling Data Extractor

## Project structure

```
data-extractor/
├── app.py              ← FastAPI application
├── schemas.py          ← Pydantic extraction schemas
├── extractor.py        ← OpenAI tool_use extraction logic
├── eval.py             ← F1 evaluation against ground truth
├── requirements.txt
├── .env.example
└── test_data/
    ├── invoices/
    ├── support_tickets/
    └── ground_truth.json
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```
# requirements.txt
openai==1.51.0
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
pymupdf==1.24.11
python-dotenv==1.0.1
httpx==0.27.2
```

```bash
# .env.example
OPENAI_API_KEY=sk-...
```

## Test data samples

Create sample unstructured texts to test extraction:

```python
# test_data/samples.py
INVOICE_TEXT = """
INVOICE #INV-2024-0892
Date: November 15, 2024
Due Date: December 15, 2024

Bill To:
Acme Corporation
123 Business Ave, Suite 100
New York, NY 10001

Description                    Qty    Unit Price    Total
Software Development Services   40    $150.00       $6,000.00
Cloud Infrastructure Setup       1    $500.00         $500.00
                                              Subtotal: $6,500.00
                                                   Tax: $520.00
                                                 Total: $7,020.00

Payment Terms: Net 30
"""

SUPPORT_TICKET_TEXT = """
From: customer@example.com
Subject: Cannot login to my account - URGENT

I've been trying to log into my account for the past 2 hours and keep getting
"Invalid credentials" even though I just reset my password. This is blocking
me from accessing critical project files before my 3pm deadline today.

Browser: Chrome 119
OS: macOS Ventura 13.5
Account: john.doe@example.com
"""
```

---

[[README]] | [[02-implementation]]
