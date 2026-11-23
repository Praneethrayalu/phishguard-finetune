"""Deterministic, label-conflict-safe cleaning and stratified splitting."""

import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DATA_URL = "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/34085a032c123ca237f314a01a67909cdea35e34/Phishing_Email.csv"
EXPECTED_SHA256 = "18ef4fff1acb8986f0ab01e83ede6025420c829656af5a377560fce17e153b97"
LABELS = {"Safe Email": 0, "Phishing Email": 1}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    return re.sub(r"\s+", " ", text).strip().lower()


def prepare(raw: Path, out: Path, seed: int = 42) -> dict:
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(DATA_URL, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with raw.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
    checksum = hashlib.sha256(raw.read_bytes()).hexdigest()
    if raw.name == "emails.csv" and checksum != EXPECTED_SHA256:
        raise ValueError(
            "Dataset checksum mismatch. Remove data/raw/emails.csv and download again."
        )
    frame = pd.read_csv(raw)
    initial = len(frame)
    frame = frame.dropna(subset=["Email Text", "Email Type"]).copy()
    frame["label"] = frame["Email Type"].map(LABELS)
    frame = frame.dropna(subset=["label"])
    frame["text"] = frame["Email Text"].map(normalize)
    frame = frame[frame.text.str.len().between(20, 100_000)].copy()
    frame["sha256"] = frame.text.map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    conflicts = frame.groupby("sha256").label.nunique()
    conflict_ids = set(conflicts[conflicts > 1].index)
    frame = frame[~frame.sha256.isin(conflict_ids)].drop_duplicates("sha256")
    frame["label"] = frame.label.astype(int)
    train, holdout = train_test_split(
        frame, test_size=0.3, stratify=frame.label, random_state=seed
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, stratify=holdout.label, random_state=seed
    )
    # Balance training only. Holdout keeps the natural class distribution.
    size = int(train.label.value_counts().min())
    train = pd.concat(
        [g.sample(size, random_state=seed) for _, g in train.groupby("label")]
    )
    train = train.sample(frac=1, random_state=seed)
    manifest = {
        "seed": seed,
        "raw_rows": initial,
        "source_url": DATA_URL,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "conflicting_hashes_removed": len(conflict_ids),
        "clean_unique_rows": len(frame),
        "normalization": "NFKC, HTML unescape, lowercase, whitespace collapse; 20..100000 chars",
        "splits": {},
    }
    seen = set()
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        assert not seen.intersection(split.sha256)
        seen.update(split.sha256)
        split[["text", "label", "sha256"]].to_json(
            out / f"{name}.jsonl", orient="records", lines=True
        )
        manifest["splits"][name] = {
            "rows": len(split),
            "labels": {str(k): int(v) for k, v in split.label.value_counts().items()},
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(root, split):
    return pd.read_json(Path(root) / f"{split}.jsonl", lines=True)
