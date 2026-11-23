import pandas as pd

from phishguard.baseline import train_baseline
from phishguard.data import prepare


def test_end_to_end_baseline(tmp_path):
    records = [
        {
            "Email Text": f"{'team meeting agenda' if label == 0 else 'urgent password verification'} number {i} do this now",
            "Email Type": "Safe Email" if label == 0 else "Phishing Email",
        }
        for label in (0, 1)
        for i in range(60)
    ]
    raw = tmp_path / "fixture.csv"
    pd.DataFrame(records).to_csv(raw, index=False)
    processed = tmp_path / "processed"
    prepare(raw, processed)
    model_path = tmp_path / "model.joblib"
    report = tmp_path / "results.json"
    result = train_baseline(processed, model_path, report)
    assert result["test"]["f1"] == 1.0
    assert model_path.is_file() and report.is_file()
