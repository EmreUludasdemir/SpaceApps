"""Train and optimise the stacking-based exoplanet classifier."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.utils import resample

from .data import PreparedDataset, load_koi_dataframe, preprocess_koi_dataframe
from .model import (
    MODEL_DESCRIPTION,
    STACKING_PARAM_GRID,
    ThresholdedClassifier,
    build_training_pipeline,
    cross_validate_pipeline,
    evaluate_thresholds,
    load_metrics,
    save_metrics,
    save_model,
    specificity_macro_scorer,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/exoplanet_classifier.joblib")
DEFAULT_METRICS_PATH = Path("models/metrics.json")
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_CV_FOLDS = 10
DEFAULT_CV_REPEATS = 5


def _balance_classes(dataset: PreparedDataset, random_state: int) -> PreparedDataset:
    frame = dataset.features.copy()
    identifiers = None
    identifier_columns: List[str] = []
    if dataset.identifiers is not None:
        identifiers = dataset.identifiers.reset_index(drop=True)
        identifier_columns = list(identifiers.columns)
        frame = pd.concat([frame.reset_index(drop=True), identifiers], axis=1)

    frame["target"] = dataset.target.reset_index(drop=True)

    grouped = []
    min_count = frame["target"].value_counts().min()
    for label, group in frame.groupby("target"):
        grouped.append(
            resample(
                group,
                replace=False,
                n_samples=min_count,
                random_state=random_state,
            )
        )

    balanced = (
        pd.concat(grouped, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    balanced_identifiers = None
    if identifier_columns:
        balanced_identifiers = balanced[identifier_columns].reset_index(drop=True)

    return PreparedDataset(
        features=balanced.drop(columns=["target", *identifier_columns]),
        target=balanced["target"].astype(int),
        numeric_features=dataset.numeric_features,
        categorical_features=dataset.categorical_features,
        label_mapping=dataset.label_mapping,
        identifiers=balanced_identifiers,
    )


def _compute_specificity_by_class(conf: np.ndarray, labels: List[str]) -> Dict[str, float]:
    totals = {}
    conf = np.asarray(conf)
    for idx, label in enumerate(labels):
        tp = conf[idx, idx]
        fp = conf[:, idx].sum() - tp
        fn = conf[idx, :].sum() - tp
        tn = conf.sum() - (tp + fp + fn)
        denom = tn + fp
        totals[label] = float(tn / denom) if denom else float("nan")
    totals["macro"] = float(np.nanmean([totals[label] for label in labels]))
    return totals


def _select_best_params(grid: GridSearchCV) -> Dict[str, object]:
    specificity_scores = grid.cv_results_["mean_test_specificity_macro"]
    best_specificity = np.max(specificity_scores)
    candidate_indices = np.flatnonzero(np.isclose(specificity_scores, best_specificity))
    if candidate_indices.size == 1:
        best_index = int(candidate_indices[0])
    else:
        f1_scores = grid.cv_results_["mean_test_f1_macro"][candidate_indices]
        best_index = int(candidate_indices[np.argmax(f1_scores)])
    return grid.cv_results_["params"][best_index]


def train(
    *,
    refresh_data: bool = False,
    random_state: int = 42,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    cv_repeats: int = DEFAULT_CV_REPEATS,
    disable_tuning: bool = False,
    reuse_tuning: bool = True,
    cv_jobs: int | None = -1,
) -> Dict[str, object]:
    start_time = time.perf_counter()

    raw_df = load_koi_dataframe(refresh=refresh_data)
    dataset = preprocess_koi_dataframe(raw_df, return_identifiers=True)
    balanced_dataset = _balance_classes(dataset, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(
        balanced_dataset.features,
        balanced_dataset.target,
        test_size=validation_size,
        random_state=random_state,
        stratify=balanced_dataset.target,
    )

    label_names = sorted(balanced_dataset.label_mapping, key=balanced_dataset.label_mapping.get)
    positive_label = balanced_dataset.label_mapping[label_names[-1]]

    pipeline = build_training_pipeline(
        numeric_features=balanced_dataset.numeric_features,
        categorical_features=balanced_dataset.categorical_features,
        random_state=random_state,
    )

    tuning_payload = None
    best_params = None

    if not disable_tuning:
        if reuse_tuning and Path(metrics_path).exists():
            try:
                cached = load_metrics(metrics_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to load cached metrics from %s: %s", metrics_path, exc)
            else:
                cached_tuning = cached.get("tuning") if isinstance(cached, dict) else None
                if isinstance(cached_tuning, dict) and cached_tuning.get("best_params"):
                    best_params = cached_tuning["best_params"]
                    tuning_payload = {**cached_tuning, "reused": True}
                    LOGGER.info("Reusing tuned parameters from %s", metrics_path)

        if best_params is None:
            scoring = {
                "specificity_macro": specificity_macro_scorer,
                "f1_macro": "f1_macro",
                "accuracy": "accuracy",
            }
            tuning_cv = RepeatedStratifiedKFold(
                n_splits=cv_folds,
                n_repeats=cv_repeats,
                random_state=random_state,
            )
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=STACKING_PARAM_GRID,
                scoring=scoring,
                cv=tuning_cv,
                n_jobs=-1,
                refit=False,
                verbose=0,
            )
            total_combinations = int(
                np.prod([len(values) for values in STACKING_PARAM_GRID.values()])
            )
            LOGGER.info(
                "Running GridSearchCV with %d combinations and %d-fold x %d-repeat CV",
                total_combinations,
                cv_folds,
                cv_repeats,
            )
            grid.fit(X_train, y_train)
            best_params = _select_best_params(grid)
            LOGGER.info("Selected tuned parameters: %s", best_params)
            tuning_payload = {
                "best_params": best_params,
                "cv_results": {
                    key: grid.cv_results_[key].tolist()
                    for key in grid.cv_results_.keys()
                    if key.startswith("mean_test")
                },
                "params": grid.cv_results_["params"],
            }
        if best_params:
            pipeline.set_params(**best_params)

    jobs = None if cv_jobs == 0 else cv_jobs

    cv = RepeatedStratifiedKFold(
        n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state
    )
    cv_payload = cross_validate_pipeline(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        positive_label=positive_label,
        n_jobs=jobs,
    )

    LOGGER.info("Fitting pipeline on training set (%d samples)", len(X_train))
    pipeline.fit(X_train, y_train)

    val_probabilities = pipeline.predict_proba(X_val)[:, positive_label]
    baseline_preds = (val_probabilities >= 0.5).astype(int)
    baseline_conf = confusion_matrix(y_val, baseline_preds, labels=list(balanced_dataset.label_mapping.values()))
    baseline_specificity = _compute_specificity_by_class(baseline_conf, label_names)

    baseline_metrics = {
        "accuracy": float(accuracy_score(y_val, baseline_preds)),
        "precision_macro": float(precision_score(y_val, baseline_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_val, baseline_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_val, baseline_preds, average="macro", zero_division=0)),
        "specificity_macro": float(baseline_specificity["macro"]),
    }

    threshold_payload = evaluate_thresholds(
        y_true=y_val.to_numpy(),
        probabilities=val_probabilities,
        positive_label=positive_label,
        baseline_f1=baseline_metrics["f1_macro"],
    )

    best_threshold = threshold_payload.get("best_threshold", 0.5)
    best_preds = (val_probabilities >= best_threshold).astype(int)
    best_conf = confusion_matrix(y_val, best_preds, labels=list(balanced_dataset.label_mapping.values()))
    best_specificity = _compute_specificity_by_class(best_conf, label_names)

    best_metrics = {
        "accuracy": float(accuracy_score(y_val, best_preds)),
        "precision_macro": float(precision_score(y_val, best_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_val, best_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_val, best_preds, average="macro", zero_division=0)),
        "specificity_macro": float(best_specificity["macro"]),
    }

    classification = classification_report(
        y_val,
        best_preds,
        labels=list(balanced_dataset.label_mapping.values()),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    metrics_summary = {
        "accuracy": best_metrics["accuracy"],
        "precision_macro": best_metrics["precision_macro"],
        "recall_macro": best_metrics["recall_macro"],
        "f1_macro": best_metrics["f1_macro"],
        "specificity_macro": best_metrics["specificity_macro"],
    }

    balanced_full = PreparedDataset(
        features=balanced_dataset.features,
        target=balanced_dataset.target,
        numeric_features=balanced_dataset.numeric_features,
        categorical_features=balanced_dataset.categorical_features,
        label_mapping=balanced_dataset.label_mapping,
        identifiers=balanced_dataset.identifiers,
    )

    LOGGER.info("Fitting final model on balanced dataset (%d samples)", len(balanced_full.features))
    pipeline.fit(balanced_full.features, balanced_full.target)

    thresholded_model = ThresholdedClassifier(
        pipeline,
        threshold=best_threshold,
        label_mapping=balanced_full.label_mapping,
    )
    save_model(thresholded_model, model_path)

    full_probabilities = thresholded_model.predict_proba(dataset.features)[:, positive_label]
    full_preds = (full_probabilities >= best_threshold).astype(int)
    label_indices = list(dataset.label_mapping.values())
    full_conf = confusion_matrix(dataset.target, full_preds, labels=label_indices)
    full_specificity = _compute_specificity_by_class(full_conf, label_names)
    full_metrics = {
        "accuracy": float(accuracy_score(dataset.target, full_preds)),
        "precision_macro": float(
            precision_score(dataset.target, full_preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(dataset.target, full_preds, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(dataset.target, full_preds, average="macro", zero_division=0)),
        "specificity_macro": float(full_specificity["macro"]),
    }

    inverse_mapping = {v: k for k, v in dataset.label_mapping.items()}
    top_k = min(25, len(full_probabilities))
    top_indices = np.argsort(full_probabilities)[::-1][:top_k]
    top_records: List[Dict[str, object]] = []
    for rank, index in enumerate(top_indices, start=1):
        probability = float(full_probabilities[index])
        predicted_label = inverse_mapping[int(full_preds[index])]
        actual_label = inverse_mapping[int(dataset.target.iloc[index])]
        record: Dict[str, object] = {
            "rank": rank,
            "index": int(index),
            "probability_candidate": probability,
            "predicted_disposition": predicted_label,
            "actual_disposition": actual_label,
        }
        feature_row = dataset.features.iloc[index]
        feature_payload: Dict[str, object] = {}
        for name, value in feature_row.items():
            if pd.isna(value):
                feature_payload[name] = None
            elif isinstance(value, (np.integer, int, np.floating, float)):
                feature_payload[name] = float(value)
            else:
                feature_payload[name] = str(value)
        record["features"] = feature_payload
        if dataset.identifiers is not None:
            for column, value in dataset.identifiers.iloc[index].items():
                if pd.isna(value):
                    record[column] = None
                elif isinstance(value, (np.integer, int)):
                    record[column] = int(value)
                elif isinstance(value, (np.floating, float)):
                    record[column] = float(value)
                elif isinstance(value, str):
                    record[column] = value
                elif hasattr(value, "item"):
                    record[column] = value.item()
                else:
                    record[column] = str(value)
        top_records.append(record)

    metrics_payload: Dict[str, object] = {
        "version": 3,
        "model_description": MODEL_DESCRIPTION,
        "metrics": metrics_summary,
        "validation": {
            "baseline_threshold": 0.5,
            "baseline_metrics": baseline_metrics,
            "best_threshold": best_threshold,
            "best_metrics": best_metrics,
            "confusion_matrix": best_conf.tolist(),
            "specificity_by_class": best_specificity,
            "classification_report": classification,
        },
        "threshold": threshold_payload,
        "cross_validation": {
            "folds": cv_folds,
            "repeats": cv_repeats,
            "per_fold": cv_payload["per_fold"],
            "summary": cv_payload["metrics"],
        },
        "tuning": tuning_payload,
        "feature_columns": {
            "numeric": balanced_full.numeric_features,
            "categorical": balanced_full.categorical_features,
        },
        "label_mapping": balanced_full.label_mapping,
        "dataset": {
            "total_records": int(len(dataset.features)),
            "balanced_records": int(len(balanced_full.features)),
            "validation_size": validation_size,
        },
        "full_dataset": {
            "metrics": full_metrics,
            "confusion_matrix": full_conf.tolist(),
            "specificity_by_class": full_specificity,
        },
        "top_predictions": {
            "positive_class": inverse_mapping[positive_label],
            "threshold": best_threshold,
            "records": top_records,
        },
    }

    metrics_payload.update(metrics_summary)

    save_metrics(metrics_payload, metrics_path)
    elapsed = time.perf_counter() - start_time
    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Training completed in %.2f seconds", elapsed)
    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true", help="Force re-download of the KOI dataset")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Where to store the trained model")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH, help="Where to store metrics JSON")
    parser.add_argument(
        "--validation-size",
        type=float,
        default=DEFAULT_VALIDATION_SIZE,
        help="Validation split ratio used for threshold optimisation",
    )
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="Number of CV folds")
    parser.add_argument("--cv-repeats", type=int, default=DEFAULT_CV_REPEATS, help="Number of CV repetitions")
    parser.add_argument("--disable-tuning", action="store_true", help="Skip GridSearchCV and use default parameters")
    parser.add_argument("--no-reuse-tuning", action="store_false", dest="reuse_tuning", help="Do not reuse cached tuned parameters from previous metrics")
    parser.add_argument("--cv-jobs", type=int, default=-1, help="Parallel jobs for cross-validation (-1 uses all cores)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    train(
        refresh_data=args.refresh_data,
        random_state=args.random_state,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        validation_size=args.validation_size,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
        disable_tuning=args.disable_tuning,
        reuse_tuning=getattr(args, "reuse_tuning", True),
        cv_jobs=args.cv_jobs,
    )


if __name__ == "__main__":
    main()
