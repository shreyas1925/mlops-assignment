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
    """Make MLflow tracking files compatible with the dockerized MLflow UI.

    Two normalisations:
    1. Rewrite absolute host paths to file:///mlruns (the path inside the mlflow container).
    2. Ensure every run meta.yaml carries both run_id and run_uuid so that older/newer
       MLflow server code paths can both parse it (RunInfo.from_dictionary requires run_uuid).
    """
    host_uri = mlruns_dir.resolve().as_uri()
    container_uri = "file:///mlruns"

    for path in mlruns_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name not in {"meta.yaml", "MLmodel"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text = text
        if host_uri != container_uri and host_uri in new_text:
            new_text = new_text.replace(host_uri, container_uri)
        if path.name == "meta.yaml":
            run_id_value = None
            has_run_uuid = False
            for line in new_text.splitlines():
                if line.startswith("run_id:") and run_id_value is None:
                    run_id_value = line.split(":", 1)[1].strip()
                if line.startswith("run_uuid:"):
                    has_run_uuid = True
            if run_id_value and not has_run_uuid:
                lines = new_text.splitlines(keepends=True)
                inserted = False
                rebuilt = []
                for line in lines:
                    rebuilt.append(line)
                    if line.startswith("run_id:") and not inserted:
                        rebuilt.append(f"run_uuid: {run_id_value}\n")
                        inserted = True
                new_text = "".join(rebuilt)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


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
