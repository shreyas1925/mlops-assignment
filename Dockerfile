FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src /app/src
COPY api /app/api

# Model artifacts are mounted at runtime (compose volume / k8s volume)
# rather than baked into the image, so the same image can serve any
# trained model without a rebuild.
RUN mkdir -p /app/artifacts/models /app/artifacts/reports /app/artifacts/mlruns

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/artifacts
USER appuser

ENV PYTHONPATH=/app/src \
    MODEL_PATH=/app/artifacts/models/heart_disease_pipeline.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
