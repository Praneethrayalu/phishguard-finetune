import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "n": len(labels),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, probabilities))
        if len(set(labels)) == 2
        else None,
        "average_precision": float(average_precision_score(labels, probabilities))
        if len(set(labels)) == 2
        else None,
        "confusion_matrix_true_rows_safe_phishing": confusion_matrix(
            labels, predicted, labels=[0, 1]
        ).tolist(),
    }


def timed_predict(predict, texts):
    start = time.perf_counter()
    probabilities = predict(texts)
    elapsed = time.perf_counter() - start
    return probabilities, {
        "total_seconds": elapsed,
        "mean_ms_per_email_batched": 1000 * elapsed / len(texts),
    }
