"""Dataset parsing and cleaning utilities for Cleveland heart disease data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

RECORD_WIDTH = 76

# Indexes are 0-based positions in the original Cleveland format.
SELECTED_FEATURE_INDEX = {
    "age": 2,
    "sex": 3,
    "cp": 8,
    "trestbps": 9,
    "chol": 11,
    "fbs": 15,
    "restecg": 18,
    "thalach": 31,
    "exang": 37,
    "oldpeak": 39,
    "slope": 40,
    "ca": 43,
    "thal": 50,
    "num": 57,
}


def _tokenize_raw_file(raw_data_path: Path) -> list[str]:
    # Cleveland raw file can include non-UTF bytes in some mirrors.
    content = raw_data_path.read_bytes().decode("latin-1", errors="ignore")
    content = content.replace("\x00", " ")
    tokens = content.split()
    if not tokens:
        raise ValueError(f"No tokens found in raw file: {raw_data_path}")
    if len(tokens) % RECORD_WIDTH != 0:
        usable_tokens = (len(tokens) // RECORD_WIDTH) * RECORD_WIDTH
        LOGGER.warning(
            "Token count (%d) is not divisible by %d, trimming to %d tokens.",
            len(tokens),
            RECORD_WIDTH,
            usable_tokens,
        )
        tokens = tokens[:usable_tokens]
    return tokens


def parse_cleveland_raw(raw_data_path: Path) -> pd.DataFrame:
    """Parse the fixed-width token stream and return selected 14 attributes."""
    tokens = _tokenize_raw_file(raw_data_path)
    record_count = len(tokens) // RECORD_WIDTH
    rows: list[dict[str, float]] = []

    for row_index in range(record_count):
        start = row_index * RECORD_WIDTH
        end = start + RECORD_WIDTH
        record = tokens[start:end]
        row: dict[str, float] = {}
        for feature_name, feature_index in SELECTED_FEATURE_INDEX.items():
            value = record[feature_index]
            if value in {"-9", "-9.0", "?"}:
                row[feature_name] = float("nan")
            else:
                try:
                    row[feature_name] = float(value)
                except ValueError:
                    row[feature_name] = float("nan")
        rows.append(row)

    parsed_df = pd.DataFrame(rows, columns=list(SELECTED_FEATURE_INDEX.keys()))
    if parsed_df.empty:
        raise ValueError("Parsed dataset is empty.")
    return parsed_df


def clean_cleveland_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean parsed dataframe and add binary target."""
    cleaned = df.copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["target"] = (cleaned["num"] > 0).astype("Int64")
    cleaned = cleaned.dropna(subset=["target"])
    cleaned["target"] = cleaned["target"].astype(int)
    return cleaned


def build_clean_dataset(raw_data_path: Path, output_path: Path) -> pd.DataFrame:
    """Parse and clean Cleveland data, then save to CSV."""
    parsed_df = parse_cleveland_raw(raw_data_path)
    cleaned_df = clean_cleveland_data(parsed_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    LOGGER.info("Parsed %d rows from %s", len(parsed_df), raw_data_path)
    LOGGER.info("Saved cleaned data to %s", output_path)
    # Required visibility log: show at least one complete object.
    LOGGER.info("Sample cleaned patient object: %s", cleaned_df.iloc[0].to_dict())
    return cleaned_df

