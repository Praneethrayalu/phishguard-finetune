"""Compare all trained models on the same untouched held-out test set."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import torch

from phishguard.data import load_split
from phishguard.metrics import evaluate, timed_predict
from phishguard.transformer import load_adapter, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["tfidf_logreg", "distilbert_lora_fp32", "distilbert_lora_int8"],
        required=True,
    )
    args = parser.parse_args()
    data = load_split("data/processed", "test")
    texts, labels = data.text.tolist(), data.label.tolist()
    baseline = joblib.load("artifacts/baseline.joblib")
    model, tokenizer = load_adapter()
    config = json.loads(Path("artifacts/lora/run.json").read_text())
    results = {
        "test_rows": len(texts),
        "seed": 42,
        "threshold": 0.5,
        "hardware": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "lora_train_rows": config["train_rows"],
        "models": {},
    }
    for name, fn in [
        ("tfidf_logreg", lambda values: baseline.predict_proba(values)[:, 1]),
        ("distilbert_lora_fp32", lambda values: predict(model, tokenizer, values)),
    ]:
        if args.model != name:
            continue
        probabilities, timing = timed_predict(fn, texts)
        results["models"][name] = {**evaluate(labels, probabilities), **timing}
    if args.model != "distilbert_lora_int8":
        Path(f"reports/{args.model}_test.json").write_text(
            json.dumps(results, indent=2)
        )
        print(json.dumps(results, indent=2))
        return
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized, "artifacts/quantized.pt")
    probabilities, timing = timed_predict(
        lambda values: predict(quantized, tokenizer, values), texts
    )
    results["models"]["distilbert_lora_int8"] = {
        **evaluate(labels, probabilities),
        **timing,
        "artifact_bytes": Path("artifacts/quantized.pt").stat().st_size,
    }
    Path(f"reports/{args.model}_test.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
