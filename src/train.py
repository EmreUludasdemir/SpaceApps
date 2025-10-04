"""Script to train the exoplanet classification model with cross-validated ensembles."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import KOI_FEATURE_COLUMNS, load_koi_dataframe, split_features_and_target
from .model import (
    MODEL_REGISTRY,
    build_model,
    cross_validate_model,
    evaluate_predictions,
    get_model_registry,
    save_metrics,
    save_model,
    tune_hyperparameters,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path("models/exoplanet_classifier.joblib")
DEFAULT_METRICS_PATH = Path("models/metrics.json")
DEFAULT_FEATURE_IMPORTANCE_PATH = Path("models/feature_importances.json")
DEFAULT_SELECTION_METRIC = "f1_macro"


def _select_best_model(cross_validation_results: Dict[str, Dict[str, object]], metric: str) -> str:
    """Return the model name with the highest mean value for the given metric."""

    best_model = None
    best_score = float("-inf")
    for name, payload in cross_validation_results.items():
        metrics = payload.get("metrics", {})
        summary = metrics.get(metric)
        if not summary:
            continue
        score = summary.get("mean", float("nan"))
        if np.isnan(score):
            continue
        if score > best_score:
            best_model = name
            best_score = score
    if best_model is None:
        raise RuntimeError(f"Unable to determine best model using metric '{metric}'.")
    return best_model


def _extract_feature_importances(pipeline, feature_importance_path: Path | str) -> None:
    """Persist feature importances when exposed by the trained classifier."""

    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        LOGGER.warning("Pipeline does not expose a classifier step; skipping feature importance export.")
        return

    feature_importances = None
    if hasattr(classifier, "feature_importances_"):
        feature_importances = classifier.feature_importances_
    elif hasattr(classifier, "estimators_") and all(
        hasattr(estimator, "feature_importances_") for estimator in getattr(classifier, "estimators_", [])
    ):
        importances = [estimator.feature_importances_ for estimator in classifier.estimators_]
        feature_importances = np.mean(importances, axis=0)

    if feature_importances is None:
        LOGGER.info("Selected classifier does not provide feature importances; skipping export.")
        return

    payload = {
        "features": list(KOI_FEATURE_COLUMNS),
        "importances": feature_importances.tolist(),
    }
    feature_path = Path(feature_importance_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Persisted feature importances to %s", feature_path)


def train(
    *,
    model_name: str = "gradient_boosting",
    random_state: int = 42,
    refresh_data: bool = False,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    feature_importance_path: Path | str = DEFAULT_FEATURE_IMPORTANCE_PATH,
    cv_splits: int = 5,
    auto_select: bool = False,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    tune: bool = False,
    tuning_metric: str = DEFAULT_SELECTION_METRIC,
    tuning_iterations: int = 10,
) -> Dict[str, object]:
    """Train the classifier, evaluate ensembles via cross-validation, and persist artefacts."""

    start_time = time.perf_counter()

    df = load_koi_dataframe(refresh=refresh_data)
    X, y = split_features_and_target(df)
    labels: Iterable[str] = sorted(y.unique())

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    candidate_models = list(MODEL_REGISTRY) if auto_select else [model_name]

    cross_validation_payload: Dict[str, Dict[str, object]] = {}
    for candidate in candidate_models:
        LOGGER.info("Evaluating %s via %d-fold stratified CV", candidate, cv_splits)
        pipeline = build_model(candidate, random_state=random_state)
        result = cross_validate_model(pipeline, X, y, cv=cv, labels=labels)
        cross_validation_payload[candidate] = {
            "description": MODEL_REGISTRY[candidate].description,
            **result,
        }

    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {sorted(MODEL_REGISTRY)}")

    selected_model = model_name
    if auto_select:
        selected_model = _select_best_model(cross_validation_payload, selection_metric)
        LOGGER.info("Auto-selected %s based on %s", selected_model, selection_metric)

    best_pipeline = build_model(selected_model, random_state=random_state)
    tuning_payload: Dict[str, object] | None = None
    if tune:
        tuning_payload = tune_hyperparameters(
            best_pipeline,
            model_name=selected_model,
            X=X,
            y=y,
            cv=cv,
            scoring=tuning_metric,
            n_iter=tuning_iterations,
            random_state=random_state,
        )
        if tuning_payload.get("enabled") and tuning_payload.get("best_params"):
            best_params = tuning_payload["best_params"]
            LOGGER.info("Applying tuned parameters for %s: %s", selected_model, best_params)
            best_pipeline.set_params(**best_params)
        else:
            LOGGER.info(
                "Hyperparameter tuning skipped for %s (reason: %s)",
                selected_model,
                tuning_payload.get("reason"),
            )

    LOGGER.info("Generating cross-validated predictions for %s", selected_model)
    cv_predictions = cross_val_predict(best_pipeline, X, y, cv=cv)
    evaluation = evaluate_predictions(y, cv_predictions, labels)

    LOGGER.info("Fitting %s on full dataset (%d samples)", selected_model, len(X))
    best_pipeline.fit(X, y)

    save_model(best_pipeline, model_path)
    _extract_feature_importances(best_pipeline, feature_importance_path)

    summary_metrics = {
        key: evaluation[key]
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro"]
        if key in evaluation
    }

    metrics_payload: Dict[str, object] = {
        "version": 2,
        "best_model": selected_model,
        "best_model_description": MODEL_REGISTRY[selected_model].description,
        "metrics": summary_metrics,
        "specificity_by_class": evaluation.get("specificity_by_class", {}),
        "classification_report": evaluation.get("classification_report"),
        "confusion_matrix": evaluation.get("confusion_matrix"),
        "labels": evaluation.get("labels"),
        "cross_validation": {
            "folds": cv_splits,
            "selection_metric": selection_metric,
            "results": cross_validation_payload,
        },
        "tuning": tuning_payload,
        "dataset_size": len(df),
        "feature_columns": list(KOI_FEATURE_COLUMNS),
    }

    # Provide backward-compatible top-level keys for Streamlit visualisations.
    metrics_payload.update(summary_metrics)

    save_metrics(metrics_payload, metrics_path)
    elapsed = time.perf_counter() - start_time
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Training completed in %.2f seconds", elapsed)
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(get_model_registry().keys()),
        default="gradient_boosting",
        help="Name of the ensemble to train.",
    )
    parser.add_argument("--refresh-data", action="store_true", help="Force re-download of the KOI dataset")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for data splitting and model")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to store the trained model")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH, help="Path to store evaluation metrics")
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=DEFAULT_FEATURE_IMPORTANCE_PATH,
        help="Path to store feature importance values",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Evaluate all registered ensembles and persist the best performer",
    )
    parser.add_argument(
        "--selection-metric",
        default=DEFAULT_SELECTION_METRIC,
        help="Metric used to determine the best model when auto-selecting",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter search for the selected model before final training",
    )
    parser.add_argument(
        "--tuning-metric",
        default=DEFAULT_SELECTION_METRIC,
        help="Metric optimised during hyperparameter search",
    )
    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=10,
        help="Number of hyperparameter combinations to evaluate when tuning",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (e.g. INFO, DEBUG)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    train(
        model_name=args.model,
        refresh_data=args.refresh_data,
        random_state=args.random_state,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        feature_importance_path=args.feature_importance_path,
        cv_splits=args.cv_splits,
        auto_select=args.auto_select,
        selection_metric=args.selection_metric,
        tune=args.tune,
        tuning_metric=args.tuning_metric,
        tuning_iterations=args.tuning_iterations,
    )


if __name__ == "__main__":
    main()
