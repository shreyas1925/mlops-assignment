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


def _make_mlruns_portable(mlruns_dir) -> None:
    """Rewrite absolute host paths in MLflow tracking files to a container-portable URI.

    MLflow records the absolute filesystem URI of the tracking dir inside meta.yaml/MLmodel
    files, which breaks the dockerized MLflow UI (it mounts mlruns at /mlruns). We rewrite
    every occurrence of the host path to file:///mlruns so the same artifacts work locally
    and inside the container without re-training.
    """
    host_uri = mlruns_dir.resolve().as_uri()
    container_uri = "file:///mlruns"
    if host_uri == container_uri:
        return
    for meta_path in mlruns_dir.rglob("*"):
        if not meta_path.is_file():
            continue
        if meta_path.name not in {"meta.yaml", "MLmodel"}:
            continue
        try:
            text = meta_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if host_uri in text:
            meta_path.write_text(text.replace(host_uri, container_uri), encoding="utf-8")


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
    _make_mlruns_portable(MLRUNS_DIR)
    logging.info("Training pipeline complete. Selected model: %s", train_results["selected_model"])


if __name__ == "__main__":
    main()
