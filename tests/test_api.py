import joblib
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from phishguard.api import create_app


def test_missing_model(tmp_path):
    with TestClient(create_app(tmp_path / "none")) as client:
        assert client.get("/health").json()["ready"] is False
        assert (
            client.post(
                "/predict", json={"text": "A normal business meeting tomorrow"}
            ).status_code
            == 503
        )


def test_prediction_and_validation(tmp_path):
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(
        [
            "meeting business tomorrow",
            "office meeting schedule",
            "password urgent prize",
            "click urgent password",
        ],
        [0, 0, 1, 1],
    )
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    with TestClient(create_app(path)) as client:
        assert client.get("/health").json()["ready"]
        response = client.post(
            "/predict", json={"text": "urgent password click prize now"}
        )
        assert response.status_code == 200
        assert 0 <= response.json()["phishing_probability"] <= 1
        for text in ("x", "x" * 100001, " " * 21):
            assert client.post("/predict", json={"text": text}).status_code == 422
