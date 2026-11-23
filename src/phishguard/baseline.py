import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import load_split
from .metrics import evaluate, timed_predict


def train_baseline(
    data="data/processed",
    output="artifacts/baseline.joblib",
    report="reports/baseline.json",
):
    train, test = load_split(data, "train"), load_split(data, "test")
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), max_features=60000, min_df=2, sublinear_tf=True
                ),
            ),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    model.fit(train.text, train.label)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    probabilities, timing = timed_predict(
        lambda x: model.predict_proba(x)[:, 1], test.text.tolist()
    )
    result = {
        "model": "TF-IDF (word 1-2 grams) + logistic regression",
        "train_n": len(train),
        "test": evaluate(test.label, probabilities),
        "inference": timing,
        "note": "Threshold fixed at 0.5; test set not used for tuning.",
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(result, indent=2))
    return result
