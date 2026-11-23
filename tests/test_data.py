import pandas as pd

from phishguard.data import normalize, prepare


def test_normalize():
    assert normalize("  A &amp; B\n C ") == "a & b c"
    assert normalize("ＡＢＣ") == "abc"


def test_prepare_no_leakage(tmp_path):
    raw = tmp_path / "input.csv"
    records = [
        {
            "Email Text": f"{'safe meeting' if label == 0 else 'phishing prize'} message unique number {i} please read",
            "Email Type": "Safe Email" if label == 0 else "Phishing Email",
        }
        for label in (0, 1)
        for i in range(60)
    ]
    records += [records[0].copy(), {**records[1], "Email Type": "Phishing Email"}]
    pd.DataFrame(records).to_csv(raw, index=False)
    output = tmp_path / "clean"
    manifest = prepare(raw, output)
    assert manifest["conflicting_hashes_removed"] == 1
    sets = [
        set(pd.read_json(output / f"{s}.jsonl", lines=True).sha256)
        for s in ("train", "validation", "test")
    ]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert len(set(manifest["splits"]["train"]["labels"].values())) == 1
    first = (output / "train.jsonl").read_bytes()
    prepare(raw, output)
    assert first == (output / "train.jsonl").read_bytes()
