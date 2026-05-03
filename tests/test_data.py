from __future__ import annotations

from heart_disease_mlops.data import clean_cleveland_data, parse_cleveland_raw
from heart_disease_mlops.settings import RAW_DATA_FILE


def test_parse_cleveland_raw_returns_expected_columns():
    parsed = parse_cleveland_raw(RAW_DATA_FILE)
    assert len(parsed) > 0
    assert parsed.columns.tolist() == [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]


def test_clean_cleveland_data_adds_binary_target():
    parsed = parse_cleveland_raw(RAW_DATA_FILE)
    cleaned = clean_cleveland_data(parsed)
    assert "target" in cleaned.columns
    assert set(cleaned["target"].unique().tolist()).issubset({0, 1})

