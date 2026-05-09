"""EDA artifact generation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def run_eda_and_save_artifacts(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate assignment-required EDA visuals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    sns.set_theme(style="whitegrid")

    # Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(x="target", data=df)
    plt.title("Heart Disease Target Class Balance")
    plt.xlabel("Target")
    plt.ylabel("Count")
    class_balance_path = output_dir / "class_balance.png"
    plt.tight_layout()
    plt.savefig(class_balance_path)
    plt.close()

    # Histogram overview
    histogram_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df[histogram_cols].hist(figsize=(14, 8), bins=20)
    plt.tight_layout()
    hist_path = output_dir / "feature_histograms.png"
    plt.savefig(hist_path)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    LOGGER.info("EDA artifacts saved to %s", output_dir)
