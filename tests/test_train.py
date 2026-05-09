from __future__ import annotations

import pandas as pd

import train


def test_train_and_select_model_outputs_files(tmp_path, monkeypatch):
    data_path = tmp_path / "train_input.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    mlruns_dir = tmp_path / "mlruns"

    rows = []
    for i in range(20):
        rows.append(
            {
                "age": 40 + i,
                "sex": i % 2,
                "cp": (i % 4) + 1,
                "trestbps": 120 + i,
                "chol": 200 + (i * 3),
                "fbs": i % 2,
                "restecg": i % 3,
                "thalach": 170 - i,
                "exang": i % 2,
                "oldpeak": float((i % 5) * 0.5),
                "slope": (i % 3) + 1,
                "ca": i % 3,
                "thal": 3 if i % 2 == 0 else 6,
                "target": i % 2,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "log_params", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train.mlflow,
        "start_run",
        lambda *args, **kwargs: _DummyContextManager(),
    )

    monkeypatch.setattr(
        train,
        "MODEL_CONFIG",
        {
            "logistic_regression": {
                "estimator": train.LogisticRegression(max_iter=300),
                "param_grid": {
                    "model__C": [1.0],
                    "model__class_weight": [None],
                },
            }
        },
    )

    result = train.train_and_select_model(
        data_path=data_path,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
        mlruns_dir=mlruns_dir,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert result["selected_model"] == "logistic_regression"
    assert "holdout_metrics" in result["logistic_regression"]


class _DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
