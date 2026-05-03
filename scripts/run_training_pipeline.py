"""End-to-end local training pipeline."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from heart_disease_mlops.settings import (
        ARTIFACTS_DIR,
        CLEAN_DATA_FILE,
        MLRUNS_DIR,
        MODEL_FILE,
        RAW_DATA_FILE,
        REPORTS_DIR,
        TRAIN_METRICS_FILE,
    )

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
    from heart_disease_mlops.data import build_clean_dataset
    from heart_disease_mlops.eda import run_eda_and_save_artifacts
    from heart_disease_mlops.train import train_and_select_model

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

