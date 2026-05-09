from __future__ import annotations

import pandas as pd

from eda import run_eda_and_save_artifacts


def test_run_eda_generates_required_artifacts(tmp_path):
    df = pd.DataFrame(
        {
            "age": [45, 52, 61, 39, 58, 47],
            "sex": [1, 0, 1, 0, 1, 1],
            "cp": [1, 2, 3, 4, 2, 1],
            "trestbps": [130, 140, 150, 120, 145, 135],
            "chol": [220, 250, 280, 210, 260, 230],
            "fbs": [0, 1, 0, 0, 1, 0],
            "restecg": [0, 1, 2, 0, 1, 2],
            "thalach": [160, 150, 140, 170, 145, 155],
            "exang": [0, 1, 1, 0, 1, 0],
            "oldpeak": [1.0, 2.3, 1.8, 0.4, 2.1, 0.9],
            "slope": [1, 2, 2, 1, 3, 1],
            "ca": [0, 1, 2, 0, 1, 0],
            "thal": [3, 6, 7, 3, 6, 3],
            "num": [0, 2, 1, 0, 3, 0],
            "target": [0, 1, 1, 0, 1, 0],
        }
    )
    run_eda_and_save_artifacts(df, tmp_path)
    assert (tmp_path / "class_balance.png").exists()
    assert (tmp_path / "feature_histograms.png").exists()
    assert (tmp_path / "correlation_heatmap.png").exists()
