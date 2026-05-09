"""End-to-end training pipeline.

Run as a module from the repo root:

    PYTHONPATH=src python -m pipeline

This orchestrates: data cleaning -> EDA artifacts -> model training + MLflow tracking.
"""

from __future__ import annotations

import logging
import os

from settings import (
    ARTIFACTS_DIR,
    CLEAN_DATA_FILE,
    MLRUNS_DIR,
    MODEL_FILE,
    RAW_DATA_FILE,
    REPORTS_DIR,
    TRAIN_METRICS_FILE,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    matplotlib_cache = ARTIFACTS_DIR / ".matplotlib"
    generic_cache = ARTIFACTS_DIR / ".cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    generic_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(generic_cache))

    from data import build_clean_dataset
    from eda import run_eda_and_save_artifacts
    from train import train_and_select_model

    cleaned_df = build_clean_dataset(RAW_DATA_FILE, CLEAN_DATA_FILE)
    run_eda_and_save_artifacts(cleaned_df, REPORTS_DIR)
    train_results = train_and_select_model(
        data_path=CLEAN_DATA_FILE,
        model_output_path=MODEL_FILE,
        metrics_output_path=TRAIN_METRICS_FILE,
        mlruns_dir=MLRUNS_DIR,
    )
    logging.info("Training pipeline complete. Selected model: %s", train_results["selected_model"])


if __name__ == "__main__":
    main()
