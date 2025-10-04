"""Model utilities and evaluation helpers for exoplanet classification."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import joblib
import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import KOI_FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing a candidate ensemble model."""

    name: str
    description: str
    requires_scaling: bool
    estimator_factory: Callable[[int], ClassifierMixin]


def _stacking_factory(random_state: int) -> StackingClassifier:
    """Return a stacked ensemble inspired by ensemble comparisons in recent literature."""

    estimators = [
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                random_state=random_state,
                learning_rate=0.05,
                n_estimators=400,
                max_depth=3,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=200,
                max_depth=None,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    final_estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "gradient_boosting": ModelSpec(
        name="gradient_boosting",
        description="Gradient Boosting classifier with median imputation and standardisation.",
        requires_scaling=True,
        estimator_factory=lambda random_state: GradientBoostingClassifier(
            random_state=random_state,
            learning_rate=0.05,
            max_depth=3,
            n_estimators=400,
        ),
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        description="Random Forest with balanced subsampling and tuned tree count.",
        requires_scaling=False,
        estimator_factory=lambda random_state: RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    ),
    "extra_trees": ModelSpec(
        name="extra_trees",
        description="Extremely Randomised Trees (ExtraTrees) with class balancing.",
        requires_scaling=False,
        estimator_factory=lambda random_state: ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    ),
    "ada_boost": ModelSpec(
        name="ada_boost",
        description="AdaBoost ensemble with shallow decision trees and moderated learning rate.",
        requires_scaling=True,
        estimator_factory=lambda random_state: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=random_state),
            n_estimators=400,
            learning_rate=0.3,
            random_state=random_state,
        ),
    ),
    "random_subspace": ModelSpec(
        name="random_subspace",
        description="Random Subspace method via Bagging with feature subsampling.",
        requires_scaling=False,
        estimator_factory=lambda random_state: BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=None, random_state=random_state),
            n_estimators=200,
            max_features=0.6,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        ),
    ),
    "stacking": ModelSpec(
        name="stacking",
        description="Stacked ensemble combining gradient boosting, random forest, and extra trees.",
        requires_scaling=True,
        estimator_factory=_stacking_factory,
    ),
}


def get_model_registry() -> Dict[str, ModelSpec]:
    """Return the registry of available ensemble configurations."""

    return MODEL_REGISTRY


MODEL_TUNING_GRID: Dict[str, Dict[str, Iterable[object]]] = {
    "gradient_boosting": {
        "classifier__learning_rate": [0.03, 0.05, 0.08],
        "classifier__n_estimators": [300, 400, 500],
        "classifier__max_depth": [2, 3, 4],
    },
    "random_forest": {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [None, 20, 40],
        "classifier__max_features": ["sqrt", "log2", 0.5],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "extra_trees": {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_depth": [None, 20, 40],
        "classifier__max_features": ["sqrt", "log2", 0.5],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "ada_boost": {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__learning_rate": [0.1, 0.3, 0.5],
        "classifier__estimator__max_depth": [2, 3, 4],
    },
    "random_subspace": {
        "classifier__n_estimators": [200, 400, 600],
        "classifier__max_features": [0.5, 0.6, 0.8],
        "classifier__estimator__max_depth": [None, 12, 24],
        "classifier__estimator__min_samples_leaf": [1, 2, 4],
    },
    "stacking": {
        "classifier__final_estimator__C": [0.3, 1.0, 3.0],
        "classifier__final_estimator__penalty": ["l2"],
        "classifier__final_estimator__solver": ["lbfgs"],
    },
}


def get_tuning_grid(model_name: str) -> Dict[str, Iterable[object]] | None:
    """Return the hyperparameter grid for a given model, if available."""

    return MODEL_TUNING_GRID.get(model_name)


def _count_param_combinations(grid: Dict[str, Iterable[object]]) -> int:
    """Return the total number of combinations in a parameter grid."""

    try:
        return len(ParameterGrid(grid))
    except Exception:  # noqa: BLE001
        # Fall back to a simple product computation.
        count = 1
        for values in grid.values():
            count *= len(list(values))
        return count


def tune_hyperparameters(
    pipeline: Pipeline,
    *,
    model_name: str,
    X,
    y,
    cv,
    scoring: str,
    n_iter: int,
    random_state: int,
) -> Dict[str, object]:
    """Perform lightweight hyperparameter search for the provided pipeline."""

    grid = get_tuning_grid(model_name)
    if not grid:
        LOGGER.info("No tuning grid defined for %s; skipping hyperparameter search.", model_name)
        return {"enabled": False, "reason": "no_grid"}

    total_combinations = _count_param_combinations(grid)
    iterations = min(max(n_iter, 1), total_combinations)
    sampler = list(ParameterSampler(grid, n_iter=iterations, random_state=random_state))

    best_params: Dict[str, object] | None = None
    best_score = float("-inf")
    evaluations = []

    LOGGER.info(
        "Running hyperparameter search for %s (%d/%d combinations sampled) using %s",
        model_name,
        iterations,
        total_combinations,
        scoring,
    )

    for index, params in enumerate(sampler, start=1):
        candidate = clone(pipeline)
        candidate.set_params(**params)
        start = time.perf_counter()
        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scoring)
        duration = float(time.perf_counter() - start)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        evaluations.append(
            {
                "params": params,
                "mean_score": mean_score,
                "std_score": std_score,
                "duration_seconds": duration,
            }
        )
        LOGGER.info(
            "Tuning candidate %d/%d: %s=%.3f (±%.3f) in %.2f seconds",
            index,
            iterations,
            scoring,
            mean_score,
            std_score,
            duration,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    payload: Dict[str, object] = {
        "enabled": True,
        "scoring": scoring,
        "iterations": iterations,
        "total_combinations": total_combinations,
        "best_params": best_params,
        "best_score": best_score,
        "evaluations": evaluations,
    }
    return payload


def build_model(model_name: str = "gradient_boosting", *, random_state: int = 42) -> Pipeline:
    """Create a scikit-learn pipeline for classifying KOI dispositions."""

    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available models: {sorted(MODEL_REGISTRY)}")

    spec = MODEL_REGISTRY[model_name]
    steps: List[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.requires_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", spec.estimator_factory(random_state)))
    return Pipeline(steps=steps)


def compute_specificity(conf: np.ndarray, labels: Iterable[str]) -> Dict[str, float]:
    """Compute specificity (true negative rate) for each class and its macro average."""

    specificity_by_class: Dict[str, float] = {}
    conf = np.asarray(conf)
    total = conf.sum()
    labels_list = list(labels)
    for idx, label in enumerate(labels_list):
        tp = conf[idx, idx]
        fp = conf[:, idx].sum() - tp
        fn = conf[idx, :].sum() - tp
        tn = total - (tp + fp + fn)
        denom = tn + fp
        specificity_by_class[label] = float(tn / denom) if denom else float("nan")
    macro = float(np.nanmean(list(specificity_by_class.values())))
    specificity_by_class["macro"] = macro
    return specificity_by_class


def summarise_basic_metrics(y_true, y_pred, labels: Iterable[str]) -> Dict[str, float]:
    """Return fundamental classification metrics for quick comparison."""

    conf = confusion_matrix(y_true, y_pred, labels=list(labels))
    specificity = compute_specificity(conf, labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": float(specificity["macro"]),
    }
    return metrics


def evaluate_predictions(y_true, y_pred, labels: Iterable[str]) -> Dict[str, object]:
    """Return a rich evaluation payload based on model predictions."""

    labels = list(labels)
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = compute_specificity(conf, labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    payload: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity_macro": float(specificity["macro"]),
        "specificity_by_class": {k: float(v) for k, v in specificity.items() if k != "macro"},
        "classification_report": report,
        "confusion_matrix": conf.tolist(),
        "labels": labels,
    }
    return payload


def cross_validate_model(
    pipeline: Pipeline,
    X,
    y,
    *,
    cv,
    labels: Iterable[str],
) -> Dict[str, object]:
    """Evaluate a pipeline using cross-validation and return fold-level metrics."""

    fold_metrics: List[Dict[str, float]] = []
    total_folds = cv.get_n_splits(X, y)
    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        cloned = clone(pipeline)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        start = time.perf_counter()
        LOGGER.info("Fitting fold %d/%d", fold_index, total_folds)
        cloned.fit(X_train, y_train)
        duration = float(time.perf_counter() - start)
        predictions = cloned.predict(X_test)
        metrics = summarise_basic_metrics(y_test, predictions, labels)
        metrics["duration_seconds"] = duration
        metrics["fold"] = fold_index
        fold_metrics.append(metrics)
        LOGGER.info(
            "Completed fold %d/%d in %.2f seconds (f1_macro=%.3f, accuracy=%.3f)",
            fold_index,
            total_folds,
            duration,
            metrics.get("f1_macro", float("nan")),
            metrics.get("accuracy", float("nan")),
        )

    aggregated: Dict[str, Dict[str, float]] = {}
    metric_keys = [key for key in fold_metrics[0] if key not in {"fold"}]
    for key in metric_keys:
        values = [fold[key] for fold in fold_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    return {"per_fold": fold_metrics, "metrics": aggregated}


def save_model(model: Pipeline, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str) -> Pipeline:
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
    "KOI_FEATURE_COLUMNS",
    "MODEL_REGISTRY",
    "MODEL_TUNING_GRID",
    "ModelSpec",
    "build_model",
    "compute_specificity",
    "cross_validate_model",
    "evaluate_predictions",
    "get_model_registry",
    "get_tuning_grid",
    "load_metrics",
    "load_model",
    "save_metrics",
    "save_model",
    "summarise_basic_metrics",
    "tune_hyperparameters",
]
