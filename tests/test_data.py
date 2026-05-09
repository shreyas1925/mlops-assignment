from __future__ import annotations

from data import (
    build_clean_dataset,
    clean_cleveland_data,
    parse_heart_disease_data,
)
from settings import RAW_DATA_FILE


def test_parse_heart_disease_data_returns_expected_columns():
    parsed = parse_heart_disease_data(RAW_DATA_FILE)
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
    parsed = parse_heart_disease_data(RAW_DATA_FILE)
    cleaned = clean_cleveland_data(parsed)
    assert "target" in cleaned.columns
    assert set(cleaned["target"].unique().tolist()).issubset({0, 1})


def test_build_clean_dataset_writes_expected_file(tmp_path):
    output_file = tmp_path / "clean.csv"
    cleaned = build_clean_dataset(RAW_DATA_FILE, output_file)
    assert output_file.exists()
    assert "target" in cleaned.columns

