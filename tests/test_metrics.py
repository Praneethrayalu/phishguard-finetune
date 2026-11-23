from phishguard.metrics import evaluate


def test_metrics():
    metrics = evaluate([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
    assert metrics["recall"] == 1
    assert metrics["precision"] == 2 / 3
    assert metrics["confusion_matrix_true_rows_safe_phishing"] == [[1, 1], [0, 2]]
