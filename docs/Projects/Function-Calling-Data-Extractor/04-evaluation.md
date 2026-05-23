# Evaluation — Function-Calling Data Extractor

## Ground truth evaluation

Build a test set with manually verified extractions. Compare predicted fields against ground truth:

```python
# ground_truth.json
[
  {
    "input": "Invoice #INV-2024-0892 from Tech Solutions Ltd, total $7,020.00 USD, due December 15, 2024",
    "schema": "invoice",
    "expected": {
      "vendor_name": "Tech Solutions Ltd",
      "invoice_number": "INV-2024-0892",
      "total_amount": 7020.0,
      "currency": "USD",
      "due_date": "2024-12-15"
    }
  }
]
```

## Field-level F1 score

```python
# eval.py
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

def field_match(predicted, expected) -> bool:
    """Fuzzy field match: handles floats, date formats, case."""
    if predicted is None and expected is None:
        return True
    if predicted is None or expected is None:
        return False
    # Float comparison
    if isinstance(expected, float):
        try:
            return abs(float(predicted) - expected) < 0.01
        except (ValueError, TypeError):
            return False
    # String comparison (case-insensitive, stripped)
    return str(predicted).strip().lower() == str(expected).strip().lower()

def evaluate_extraction(predicted: dict, expected: dict) -> dict:
    """Compute precision, recall, F1 at the field level."""
    expected_fields = {k: v for k, v in expected.items() if v is not None}
    true_positives = sum(
        1 for field, val in expected_fields.items()
        if field_match(predicted.get(field), val)
    )
    false_negatives = len(expected_fields) - true_positives
    predicted_non_null = {k: v for k, v in predicted.items()
                          if v is not None and k not in {"confidence", "validation_errors"}}
    false_positives = sum(
        1 for field in predicted_non_null
        if field not in expected_fields
    )

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": true_positives}

def run_evaluation(ground_truth_path: str = "test_data/ground_truth.json",
                   base_url: str = "http://localhost:8000") -> dict:
    with open(ground_truth_path) as f:
        test_cases = json.load(f)

    all_f1 = []
    with httpx.Client(timeout=30.0) as client:
        for case in test_cases:
            resp = client.post(f"{base_url}/extract/text", json={
                "text": case["input"],
                "schema_name": case["schema"],
            })
            if resp.status_code != 200:
                print(f"FAIL: {case['input'][:40]} → {resp.status_code}")
                continue

            predicted = resp.json()
            metrics = evaluate_extraction(predicted, case["expected"])
            all_f1.append(metrics["f1"])

            print(f"  F1={metrics['f1']:.2f} P={metrics['precision']:.2f} R={metrics['recall']:.2f} | {case['input'][:50]}")

    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
    print(f"\n=== Extraction Evaluation ===")
    print(f"Test cases:  {len(test_cases)}")
    print(f"Average F1:  {avg_f1:.2f}")
    return {"avg_f1": avg_f1, "n": len(all_f1)}

if __name__ == "__main__":
    run_evaluation()
```

## Confidence calibration

A well-calibrated confidence score should correlate with actual accuracy. Check this:

```python
def check_confidence_calibration(results: list[dict]) -> None:
    """Results: list of {confidence, f1} dicts."""
    high_conf = [r for r in results if r["confidence"] >= 0.8]
    low_conf = [r for r in results if r["confidence"] < 0.8]

    avg_f1_high = sum(r["f1"] for r in high_conf) / len(high_conf) if high_conf else 0
    avg_f1_low = sum(r["f1"] for r in low_conf) / len(low_conf) if low_conf else 0

    print(f"High confidence (≥0.8): avg F1 = {avg_f1_high:.2f} ({len(high_conf)} cases)")
    print(f"Low confidence (<0.8):  avg F1 = {avg_f1_low:.2f} ({len(low_conf)} cases)")
    print(f"Calibration gap: {avg_f1_high - avg_f1_low:.2f} (higher = better calibrated)")
```

> [!success] Target metrics
> - Average field-level F1 ≥ 0.80
> - Confidence calibration gap ≥ 0.15 (high-confidence extractions should score 0.15+ higher than low-confidence)
> - Schema coverage: < 10% of fields return `null` for clearly present information

---

[[03-advanced-features]] | [[05-deployment]]
