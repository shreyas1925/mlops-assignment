"""FastAPI model serving application."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from settings import MODEL_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("heart-disease-api")

REQUEST_COUNTER = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds")


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float


def _load_model():
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_FILE)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Train first with: PYTHONPATH=src python -m pipeline"
        )
    LOGGER.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def _to_feature_frame(request_data: PredictionRequest):
    import pandas as pd

    expected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    row = request_data.model_dump()
    return pd.DataFrame([row], columns=expected_columns)


def create_app():
    app = FastAPI(title="Heart Disease Predictor API", version="1.0.0")
    model = _load_model()

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        REQUEST_COUNTER.labels(
            endpoint=request.url.path, method=request.method, status=str(response.status_code)
        ).inc()
        LOGGER.info(
            "Request path=%s method=%s status=%s duration=%.4fs",
            request.url.path,
            request.method,
            response.status_code,
            duration,
        )
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest):
        try:
            start = time.time()
            frame = _to_feature_frame(payload)
            prediction = int(model.predict(frame)[0])
            confidence = float(model.predict_proba(frame)[0][prediction])
            PREDICTION_LATENCY.observe(time.time() - start)
            return PredictionResponse(prediction=prediction, confidence=confidence)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return app


if os.getenv("SKIP_APP_INIT") == "1":
    app = FastAPI(title="Heart Disease Predictor API", version="1.0.0")
else:
    app = create_app()

