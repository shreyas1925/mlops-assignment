"""Model training and experiment tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor

LOGGER = logging.getLogger(__name__)

MODEL_CONFIG = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000),
        "param_grid": {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
        },
    },
}


def _build_base_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def _evaluate_on_holdout(
    best_estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    y_pred = best_estimator.predict(x_test)
    y_pred_proba = best_estimator.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }


def train_and_select_model(
    data_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    mlruns_dir: Path,
) -> dict:
    """Train multiple models with MLflow tracking and save the best one."""
    df = pd.read_csv(data_path)
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = df[feature_columns]
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment("heart-disease-uci")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results: dict[str, dict] = {}

    for model_name, model_details in MODEL_CONFIG.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = _build_base_pipeline(model_details["estimator"])
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=model_details["param_grid"],
                scoring="roc_auc",
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            search.fit(x_train, y_train)

            holdout_metrics = _evaluate_on_holdout(search.best_estimator_, x_test, y_test)
            cv_roc_auc = float(search.best_score_)

            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_roc_auc", cv_roc_auc)
            for metric_name, metric_value in holdout_metrics.items():
                mlflow.log_metric(f"holdout_{metric_name}", metric_value)

            mlflow.sklearn.log_model(
                sk_model=search.best_estimator_,
                name="model",
                input_example=x_train.head(1),
                registered_model_name=None,
            )

            all_results[model_name] = {
                "best_params": search.best_params_,
                "cv_roc_auc": cv_roc_auc,
                "holdout_metrics": holdout_metrics,
                "best_estimator": search.best_estimator_,
            }
            LOGGER.info(
                "Model %s complete | CV ROC-AUC=%.4f | Holdout ROC-AUC=%.4f",
                model_name,
                cv_roc_auc,
                holdout_metrics["roc_auc"],
            )

    best_model_name = max(all_results, key=lambda name: all_results[name]["cv_roc_auc"])
    best_bundle = all_results[best_model_name]
    best_estimator = best_bundle["best_estimator"]

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, model_output_path)

    serializable_results = {
        name: {
            "best_params": result["best_params"],
            "cv_roc_auc": result["cv_roc_auc"],
            "holdout_metrics": result["holdout_metrics"],
        }
        for name, result in all_results.items()
    }
    serializable_results["selected_model"] = best_model_name

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    LOGGER.info("Saved best model (%s) to %s", best_model_name, model_output_path)
    return serializable_results
