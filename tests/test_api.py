from __future__ import annotations

import os

from fastapi.testclient import TestClient

os.environ["SKIP_APP_INIT"] = "1"

from api.main import create_app


class DummyModel:
    def predict(self, frame):
        return [1]

    def predict_proba(self, frame):
        return [[0.2, 0.8]]


def test_predict_endpoint_returns_prediction(monkeypatch):
    monkeypatch.setattr("api.main._load_model", lambda: DummyModel())
    app = create_app()
    client = TestClient(app)
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0 <= body["confidence"] <= 1

