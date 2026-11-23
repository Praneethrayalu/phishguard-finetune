"""Run locally generated int8 model. Never use an untrusted pickle."""

import sys

import torch
from transformers import AutoTokenizer

from phishguard.data import normalize
from phishguard.transformer import predict, seed_all

seed_all()
model = torch.load("artifacts/quantized.pt", map_location="cpu", weights_only=False)
tokenizer = AutoTokenizer.from_pretrained("artifacts/lora", local_files_only=True)
text = normalize(sys.stdin.read())
if not 20 <= len(text) <= 100000:
    raise SystemExit("Provide 20..100000 characters of email text on stdin")
probability = predict(model, tokenizer, [text])[0]
print(
    {
        "phishing_probability": probability,
        "label": "phishing" if probability >= 0.5 else "safe",
        "warning": "Experimental, not a safety guarantee",
    }
)
