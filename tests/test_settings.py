from __future__ import annotations

from settings import CLEAN_DATA_FILE, MODEL_FILE, RAW_DATA_FILE


def test_default_paths_are_well_formed():
    assert RAW_DATA_FILE.exists()
    assert RAW_DATA_FILE.name == "processed.cleveland.data"
    assert CLEAN_DATA_FILE.name == "processed_cleveland_clean.csv"
    assert MODEL_FILE.name == "heart_disease_pipeline.joblib"
