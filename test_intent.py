"""
Scratch script for testing intent classification approaches against the
candidate label set the app uses. Two paths are compared:

  1. Hugging Face Inference API (remote, requires HF_API_TOKEN)
  2. Local BART-MNLI via transformers pipeline (what app.py actually uses)

Run either block directly; this is a notebook port, not a test suite.
"""

import os
import requests
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

CANDIDATE_LABELS = [
    "account",
    "home",
    "products",
    "cart",
    "product's price high to low",
    "product's price low to high",
    "add to cart",
    "remove from cart",
    "product's details information",
]


# ---------------------------------------------------------------------------
# Path 1: Hugging Face Inference API (remote)
# ---------------------------------------------------------------------------
def classify_via_api(text: str) -> str:
    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise RuntimeError("Set HF_API_TOKEN env var to use the Inference API")

    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": CANDIDATE_LABELS},
    }
    output = requests.post(url, headers=headers, json=payload).json()
    max_index = output["scores"].index(max(output["scores"]))
    return output["labels"][max_index]


# ---------------------------------------------------------------------------
# Path 2: Local pipeline (what app.py uses)
# ---------------------------------------------------------------------------
def classify_local(text: str):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    )
    classifier = pipeline(
        "zero-shot-classification", model=model, tokenizer=tokenizer
    )
    result = classifier(text, CANDIDATE_LABELS)
    return result["labels"][0], result["scores"][0]


if __name__ == "__main__":
    label, confidence = classify_local("show me my details")
    print(f"Classified Label: {label}")
    print(f"Label Confidence: {confidence:.4f}")
