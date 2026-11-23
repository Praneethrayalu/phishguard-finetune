"""Never logs email bodies; no external requests during classification."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import normalize


class EmailRequest(BaseModel):
    text: str = Field(min_length=20, max_length=100_000)


class Verdict(BaseModel):
    label: str
    phishing_probability: float
    model: str
    warning: str = "Experimental signal, not a safety guarantee. Never open a link just because this says safe."


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(
            model_path
            or os.environ.get("PHISHGUARD_MODEL", "artifacts/baseline.joblib")
        )
        # Untrusted joblib files can execute code. Use your own locally trained model.
        app.state.model = joblib.load(path) if path.is_file() else None
        yield
        app.state.model = None

    app = FastAPI(title="PhishGuard", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": app.state.model is not None, "model": "tfidf-logreg"}

    @app.post("/predict", response_model=Verdict)
    def predict(request: EmailRequest):
        if app.state.model is None:
            raise HTTPException(503, "Train the baseline first: phishguard baseline")
        text = normalize(request.text)
        if len(text) < 20:
            raise HTTPException(422, "Email is too short after normalization")
        probability = float(app.state.model.predict_proba([text])[0, 1])
        return Verdict(
            label="phishing" if probability >= 0.5 else "safe",
            phishing_probability=probability,
            model="tfidf-logreg",
        )

    return app


app = create_app()
