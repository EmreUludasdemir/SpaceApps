"""Model utilities and evaluation helpers for exoplanet classification."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

LOGGER = logging.getLogger(__name__)

MODEL_DESCRIPTION = (
    "Stacking ensemble combining Random Forest, Extra Trees, and XGBoost with a logistic "
    "regression meta-learner."
)


STACKING_PARAM_GRID: Dict[str, List[object]] = {
    "classifier__random_forest__n_estimators": [300, 500],
    "classifier__random_forest__max_depth": [None, 30],
    "classifier__extra_trees__n_estimators": [300, 500],
    "classifier__extra_trees__max_depth": [None, 30],
    "classifier__xgb__n_estimators": [400, 600],
    "classifier__xgb__max_depth": [4, 6],
    "classifier__xgb__learning_rate": [0.05, 0.1],
    "classifier__final_estimator__C": [0.5, 1.0, 2.0],
}


def build_preprocessor(
    *, numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> ColumnTransformer:
    """Create a preprocessing column transformer for numeric and categorical data."""

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, list(numeric_features)))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, list(categorical_features)))

    if not transformers:
        raise ValueError("Preprocessor requires at least one feature column.")

    return ColumnTransformer(transformers=transformers)


def build_classifier(random_state: int = 42) -> StackingClassifier:
    """Construct the stacking ensemble specified in the optimisation prompt."""

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    extra_trees = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.1,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    final_estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )

    return StackingClassifier(
        estimators=[
            ("random_forest", random_forest),
            ("extra_trees", extra_trees),
            ("xgb", xgb),
        ],
        final_estimator=final_estimator,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def build_training_pipeline(
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int = 42,
) -> Pipeline:
    """Return a preprocessing + stacking pipeline ready for training."""

    preprocessor = build_preprocessor(
        numeric_features=numeric_features, categorical_features=categorical_features
    )
    classifier = build_classifier(random_state=random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def compute_specificity(tn: float, fp: float) -> float:
    """Compute specificity (true negative rate)."""

    denom = tn + fp
    return float(tn / denom) if denom else float("nan")


def macro_specificity(conf: np.ndarray) -> float:
    """Return the macro average specificity for a confusion matrix."""

    conf = np.asarray(conf)
    specificities = []
    for idx in range(conf.shape[0]):
        tp = conf[idx, idx]
        fp = conf[:, idx].sum() - tp
        fn = conf[idx, :].sum() - tp
        tn = conf.sum() - (tp + fp + fn)
        specificities.append(compute_specificity(tn, fp))
    return float(np.nanmean(specificities))


def specificity_macro_score(y_true, y_pred) -> float:
    """Scikit-learn compatible macro specificity scorer."""

    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    return macro_specificity(conf)


specificity_macro_scorer = make_scorer(specificity_macro_score)


def collect_fold_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int,
) -> Dict[str, float]:
    """Return binary classification diagnostics for a single fold."""

    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
    specificity = compute_specificity(tn, fp)
    sensitivity = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "recall": float(sensitivity),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def aggregate_metrics(per_fold: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean and standard deviation for recorded fold metrics."""

    summary: Dict[str, Dict[str, float]] = {}
    keys = per_fold[0].keys()
    for key in keys:
        values = [fold[key] for fold in per_fold]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def cross_validate_pipeline(
    pipeline: Pipeline,
    X,
    y,
    *,
    cv,
    positive_label: int,
    n_jobs: int | None = None,
) -> Dict[str, object]:
    """Evaluate the pipeline with repeated stratified CV and log per-fold metrics."""

    splits = list(enumerate(cv.split(X, y), start=1))

    def _run_fold(fold_index: int, train_idx, test_idx) -> Dict[str, float]:
        LOGGER.info("Fitting fold %d", fold_index)
        model = clone(pipeline)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        start = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.perf_counter() - start

        metrics = collect_fold_metrics(y_true=y_test, y_pred=y_pred, positive_label=positive_label)
        conf = confusion_matrix(y_test, y_pred, labels=[0, 1])
        metrics["macro_specificity"] = macro_specificity(conf)
        metrics["macro_f1"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["macro_precision"] = precision_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["macro_recall"] = recall_score(y_test, y_pred, average="macro", zero_division=0)
        metrics["fold"] = fold_index
        metrics["elapsed_seconds"] = float(elapsed)
        LOGGER.info("Completed fold %d in %.2fs", fold_index, elapsed)
        return metrics

    results: List[Dict[str, float]] = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_fold)(fold_index, train_idx, test_idx)
        for fold_index, (train_idx, test_idx) in splits
    )

    per_fold = sorted(results, key=lambda item: item["fold"])

    def _collect(metric: str) -> List[float]:
        return [fold[metric] for fold in per_fold]

    all_accuracy = _collect("accuracy")
    all_macro_specificity = _collect("macro_specificity")
    all_macro_f1 = _collect("macro_f1")
    all_macro_precision = _collect("macro_precision")
    all_macro_recall = _collect("macro_recall")

    summary = {
        "accuracy": {
            "mean": float(np.mean(all_accuracy)),
            "std": float(np.std(all_accuracy, ddof=1)) if len(all_accuracy) > 1 else 0.0,
        },
        "specificity_macro": {
            "mean": float(np.mean(all_macro_specificity)),
            "std": float(np.std(all_macro_specificity, ddof=1)) if len(all_macro_specificity) > 1 else 0.0,
        },
        "f1_macro": {
            "mean": float(np.mean(all_macro_f1)),
            "std": float(np.std(all_macro_f1, ddof=1)) if len(all_macro_f1) > 1 else 0.0,
        },
        "precision_macro": {
            "mean": float(np.mean(all_macro_precision)),
            "std": float(np.std(all_macro_precision, ddof=1)) if len(all_macro_precision) > 1 else 0.0,
        },
        "recall_macro": {
            "mean": float(np.mean(all_macro_recall)),
            "std": float(np.std(all_macro_recall, ddof=1)) if len(all_macro_recall) > 1 else 0.0,
        },
    }

    return {"per_fold": per_fold, "metrics": summary}


def evaluate_thresholds(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    positive_label: int,
    baseline_f1: float,
    thresholds: Iterable[float] | None = None,
) -> Dict[str, object]:
    """Grid-search decision thresholds to maximise specificity under an F1 constraint."""

    probabilities = np.asarray(probabilities)
    if thresholds is None:
        dense_grid = np.linspace(0.01, 0.99, 199)
        probability_points = np.unique(np.round(probabilities, 6))
        thresholds = np.unique(
            np.clip(np.concatenate([dense_grid, probability_points]), 0.0, 1.0)
        )
    else:
        thresholds = np.unique(np.clip(np.asarray(list(thresholds), dtype=float), 0.0, 1.0))

    best_threshold = 0.5
    best_specificity = float("-inf")
    best_payload: Dict[str, float] | None = None

    records: List[Dict[str, float]] = []
    baseline = max(0.0, baseline_f1)

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        conf = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = conf[0, 0], conf[0, 1], conf[1, 0], conf[1, 1]
        specificity = compute_specificity(tn, fp)
        precision = precision_score(y_true, preds, pos_label=positive_label, zero_division=0)
        recall = recall_score(y_true, preds, pos_label=positive_label, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=positive_label, zero_division=0)
        macro_spec = macro_specificity(conf)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        record = {
            "threshold": float(threshold),
            "specificity_macro": float(macro_spec),
            "f1_macro": float(macro_f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
        }
        records.append(record)

        if macro_f1 >= baseline and macro_spec > best_specificity:
            best_specificity = macro_spec
            best_threshold = float(threshold)
            best_payload = record

    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, probabilities)

    return {
        "best_threshold": best_threshold,
        "best_payload": best_payload,
        "candidates": records,
        "precision_recall_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
            "thresholds": thresholds_curve.tolist(),
        },
    }


class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper adding a decision threshold and label decoding to a pipeline."""

    def __init__(self, pipeline: Pipeline, *, threshold: float, label_mapping: Dict[str, int]):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.label_mapping = dict(label_mapping)
        self.inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        self.classes_ = np.array(sorted(self.label_mapping, key=self.label_mapping.get))

    def fit(self, X, y):  # pragma: no cover - wrapper maintains fitted pipeline
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        probabilities = self.predict_proba(X)
        positive_index = self.label_mapping[max(self.label_mapping, key=self.label_mapping.get)]
        preds = (probabilities[:, positive_index] >= self.threshold).astype(int)
        return np.array([self.inverse_mapping[int(label)] for label in preds])

    def predict_proba(self, X):
        proba = self.pipeline.predict_proba(X)
        ordered = np.zeros_like(proba)
        for idx, label in enumerate(self.pipeline.classes_):
            class_name = self.inverse_mapping[int(label)]
            target_idx = int(self.label_mapping[class_name])
            ordered[:, target_idx] = proba[:, idx]
        return ordered

    def decision_function(self, X):  # pragma: no cover - compatibility helper
        if hasattr(self.pipeline, "decision_function"):
            return self.pipeline.decision_function(X)
        raise AttributeError("Underlying pipeline does not expose decision_function")


def save_model(model: ThresholdedClassifier, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str) -> ThresholdedClassifier:
    return joblib.load(path)


def save_metrics(metrics: Dict[str, object], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    return path


def load_metrics(path: Path | str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


__all__ = [
    "MODEL_DESCRIPTION",
    "STACKING_PARAM_GRID",
    "ThresholdedClassifier",
    "aggregate_metrics",
    "build_classifier",
    "build_preprocessor",
    "build_training_pipeline",
    "collect_fold_metrics",
    "cross_validate_pipeline",
    "evaluate_thresholds",
    "load_metrics",
    "load_model",
    "macro_specificity",
    "save_metrics",
    "save_model",
    "specificity_macro_scorer",
]
