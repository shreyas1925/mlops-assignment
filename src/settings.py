"""Central project settings and file paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_ARTIFACTS_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MLRUNS_DIR = ARTIFACTS_DIR / "mlruns"

default_raw_data = DATA_DIR / "processed.cleveland.data"
configured_raw_data = Path(os.getenv("RAW_DATA_FILE", str(default_raw_data)))
RAW_DATA_FILE = configured_raw_data if configured_raw_data.exists() else default_raw_data
CLEAN_DATA_FILE = Path(
    os.getenv("CLEAN_DATA_FILE", str(DATA_ARTIFACTS_DIR / "processed_cleveland_clean.csv"))
)
TRAIN_METRICS_FILE = Path(
    os.getenv("TRAIN_METRICS_FILE", str(REPORTS_DIR / "training_metrics.json"))
)
MODEL_FILE = Path(
    os.getenv("MODEL_FILE", str(MODELS_DIR / "heart_disease_pipeline.joblib"))
)

print("Added log for demo")
