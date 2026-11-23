"""CPU-friendly DistilBERT LoRA. No paid APIs. Explicit run budgets."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load_split
from .metrics import evaluate

BASE = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def predict(model, tokenizer, texts, batch_size=16, max_length=128):
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            values.extend(model(**tokens).logits.softmax(-1)[:, 1].tolist())
    return values


def train_lora(
    data="data/processed",
    output="artifacts/lora",
    max_samples=0,
    epochs=1,
    max_length=128,
    batch_size=8,
    eval_samples=256,
):
    seed_all()
    train = load_split(data, "train")
    validation = load_split(data, "validation")
    if max_samples:
        # Keep the limited run class-balanced too.
        n = min(max_samples // 2, int(train.label.value_counts().min()))
        train = (
            train.groupby("label", group_keys=False)
            .sample(n=n, random_state=42)
            .sample(frac=1, random_state=42)
        )
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE,
        revision=REVISION,
        num_labels=2,
        id2label={0: "safe", 1: "phishing"},
        label2id={"safe": 0, "phishing": 1},
    )
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"],
            modules_to_save=["pre_classifier", "classifier"],
        ),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01
    )
    losses = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        shuffled = train.sample(frac=1, random_state=42 + epoch)
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled.iloc[i : i + batch_size]
            tokens = tokenizer(
                batch.text.tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                **tokens, labels=torch.tensor(batch.label.tolist(), dtype=torch.long)
            ).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) % 16 == 0:
                print(json.dumps({"step": len(losses), "loss": losses[-1]}), flush=True)
    elapsed = time.perf_counter() - started
    Path(output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    subset = validation.sample(min(eval_samples, len(validation)), random_state=42)
    metrics = evaluate(
        subset.label,
        predict(model, tokenizer, subset.text.tolist(), max_length=max_length),
    )
    result = {
        "base_model": BASE,
        "train_rows": len(train),
        "epochs": epochs,
        "steps": len(losses),
        "max_length": max_length,
        "batch_size": batch_size,
        "training_seconds": elapsed,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "last_loss": losses[-1],
        "validation": metrics,
        "test_evaluated": False,
        "warning": "Limited CPU run is not evidence of production quality. Baseline and LoRA results on different sets are not comparable.",
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/lora.json").write_text(json.dumps(result, indent=2))
    (Path(output) / "run.json").write_text(json.dumps(result, indent=2))
    return result


def load_adapter(path="artifacts/lora"):
    seed_all()
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=REVISION, num_labels=2
    )
    model = PeftModel.from_pretrained(base, path).merge_and_unload()
    return model.eval(), AutoTokenizer.from_pretrained(path)


def quantize(
    path="artifacts/lora",
    output="artifacts/quantized.pt",
    data="data/processed",
    eval_samples=128,
):
    model, tokenizer = load_adapter(path)
    subset = load_split(data, "validation").sample(
        n=min(eval_samples, len(load_split(data, "validation"))), random_state=42
    )
    before = predict(model, tokenizer, subset.text.tolist())
    # Dynamic int8 linear layers run on CPU. Artifact is trusted local pickle only.
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    start = time.perf_counter()
    after = predict(quantized, tokenizer, subset.text.tolist())
    seconds = time.perf_counter() - start
    torch.save(quantized, output)
    report = {
        "backend": "PyTorch dynamic int8 Linear (CPU)",
        "validation": evaluate(subset.label, after),
        "max_probability_delta_from_fp32": max(
            abs(a - b) for a, b in zip(before, after)
        ),
        "artifact_bytes": Path(output).stat().st_size,
        "inference_seconds": seconds,
        "security": "Load only locally generated trusted artifacts; torch pickle can execute code.",
    }
    Path("reports/quantization.json").write_text(json.dumps(report, indent=2))
    return report
