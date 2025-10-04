"""Streamlit application for interacting with the exoplanet classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data import KOI_FEATURE_COLUMNS, TARGET_COLUMN, load_koi_dataframe
from src.model import load_metrics, load_model

MODEL_PATH = PROJECT_ROOT / "models" / "exoplanet_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "models" / "feature_importances.json"

st.set_page_config(page_title="Exoplanet Transit Classifier", layout="wide")
st.title("Exoplanet Transit Classification Toolkit")


@st.cache_data(show_spinner=True)
def get_dataset() -> pd.DataFrame:
    return load_koi_dataframe()


@st.cache_resource(show_spinner=True)
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metrics() -> Dict[str, object] | None:
    if METRICS_PATH.exists():
        return load_metrics(METRICS_PATH)
    return None


@st.cache_data(show_spinner=False)
def get_feature_importances() -> pd.DataFrame | None:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return None
    payload = json.loads(FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
    return pd.DataFrame({"feature": payload["features"], "importance": payload["importances"]})


def _extract_summary_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    if not metrics:
        return {}
    if "metrics" in metrics and isinstance(metrics["metrics"], dict):
        return metrics["metrics"]
    return {
        key: metrics.get(key)
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro"]
        if key in metrics
    }


def _build_cv_dataframe(metrics: Dict[str, object]) -> pd.DataFrame:
    cross_val = metrics.get("cross_validation", {}) if metrics else {}
    results = cross_val.get("results", {}) if isinstance(cross_val, dict) else {}
    rows = []
    for name, payload in results.items():
        row = {
            "Model": name,
            "Description": payload.get("description", ""),
        }
        summary = payload.get("metrics", {})
        for metric_name in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro"]:
            metric_stats = summary.get(metric_name)
            if isinstance(metric_stats, dict):
                row[f"{metric_name} (mean)"] = metric_stats.get("mean")
                row[f"{metric_name} (std)"] = metric_stats.get("std")
        per_fold = payload.get("per_fold", [])
        if per_fold:
            durations = [fold.get("duration_seconds") for fold in per_fold if fold.get("duration_seconds") is not None]
            if durations:
                row["Avg. fold time (s)"] = float(np.mean(durations))
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "f1_macro (mean)" in df.columns:
        df = df.sort_values(by="f1_macro (mean)", ascending=False, na_position="last")
    return df


def render_metrics_section(metrics: Dict[str, object] | None) -> None:
    st.header("Model performance")
    if not metrics:
        st.warning("No metrics found. Run `python -m src.train` to train the model and populate metrics.")
        return

    summary = _extract_summary_metrics(metrics)
    cols = st.columns(3)
    if summary.get("accuracy") is not None:
        cols[0].metric("Cross-validated accuracy", f"{summary['accuracy']:.3f}")
    if summary.get("f1_macro") is not None:
        cols[1].metric("Macro F1", f"{summary['f1_macro']:.3f}")
    if summary.get("specificity_macro") is not None:
        cols[2].metric("Macro specificity", f"{summary['specificity_macro']:.3f}")

    best_model = metrics.get("best_model")
    if best_model:
        description = metrics.get("best_model_description") or ""
        st.markdown(f"**Selected model:** `{best_model}` — {description}")

    tuning = metrics.get("tuning") if isinstance(metrics, dict) else None
    if tuning and tuning.get("enabled"):
        score = tuning.get("best_score")
        iterations = tuning.get("iterations")
        total = tuning.get("total_combinations")
        scoring = tuning.get("scoring")
        st.markdown(
            f"**Hyperparameter tuning:** optimised `{scoring}` over {iterations}/{total} sampled combinations."
        )
        if isinstance(score, (int, float)):
            st.caption(f"Best mean CV score: {score:.3f}")
        best_params = tuning.get("best_params")
        if isinstance(best_params, dict) and best_params:
            st.json(best_params)

    class_report = metrics.get("classification_report")
    if class_report:
        class_report_df = pd.DataFrame(class_report).T
        rename_map = {"precision": "Precision", "recall": "Recall", "f1-score": "F1-score"}
        class_report_df = class_report_df.rename(columns=rename_map)
        st.dataframe(
            class_report_df.style.format({col: "{:.2f}" for col in ["Precision", "Recall", "F1-score"] if col in class_report_df}),
            use_container_width=True,
        )

    specificity_by_class = metrics.get("specificity_by_class")
    if specificity_by_class:
        spec_df = (
            pd.Series(specificity_by_class)
            .rename("Specificity")
            .to_frame()
            .reset_index()
            .rename(columns={"index": "Disposition"})
        )
        spec_chart = (
            alt.Chart(spec_df)
            .mark_bar()
            .encode(x="Specificity:Q", y=alt.Y("Disposition:N", sort="-x"))
            .properties(title="Specificity by class", width=400, height=200)
        )
        st.altair_chart(spec_chart, use_container_width=False)

    matrix = np.array(metrics.get("confusion_matrix") or [])
    labels = metrics.get("labels", [])
    if matrix.size and labels:
        matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
        heatmap = (
            alt.Chart(matrix_df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count"))
            .mark_rect()
            .encode(x="Predicted:O", y="index:O", color="Count:Q", tooltip=["index", "Predicted", "Count"])
            .properties(width=300, height=300, title="Confusion matrix")
        )
        st.altair_chart(heatmap, use_container_width=False)

    feature_df = get_feature_importances()
    if feature_df is not None and not feature_df.empty:
        chart = (
            alt.Chart(feature_df)
            .mark_bar()
            .encode(
                x="importance:Q",
                y=alt.Y("feature:N", sort="-x"),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(title="Feature importance", width=400, height=300)
        )
        st.altair_chart(chart, use_container_width=False)

    cv_df = _build_cv_dataframe(metrics)
    if not cv_df.empty:
        st.subheader("Cross-validation comparison")
        selection_metric = metrics.get("cross_validation", {}).get("selection_metric") if metrics else None
        if selection_metric:
            st.caption(f"Models ranked by `{selection_metric}`.")
        styled = cv_df.style.format("{:.3f}", subset=[col for col in cv_df.columns if col.endswith(("(mean)", "(std)", "time (s)"))])
        st.dataframe(styled, use_container_width=True)


def render_dataset_section(df: pd.DataFrame) -> None:
    st.header("Explore the Kepler Objects of Interest dataset")
    with st.expander("Dataset preview", expanded=True):
        disposition_filter = st.multiselect(
            "Filter by disposition", sorted(df[TARGET_COLUMN].unique()), default=list(sorted(df[TARGET_COLUMN].unique()))
        )
        filtered = df[df[TARGET_COLUMN].isin(disposition_filter)]
        st.write(f"Showing {len(filtered):,} of {len(df):,} records.")
        st.dataframe(filtered.head(100))

    with st.expander("Feature distributions"):
        feature = st.selectbox("Select feature", KOI_FEATURE_COLUMNS)
        chart = (
            alt.Chart(df)
            .mark_bar(opacity=0.7)
            .encode(x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=40)), y="count()", color=f"{TARGET_COLUMN}:N")
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


def _infer_bounds(df: pd.DataFrame, feature: str) -> tuple[float, float]:
    series = df[feature].dropna()
    if series.empty:
        return (0.0, 1.0)
    lower = float(series.quantile(0.01))
    upper = float(series.quantile(0.99))
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def render_manual_prediction(df: pd.DataFrame) -> None:
    st.subheader("Manual input")
    model = None
    if MODEL_PATH.exists():
        model = get_model()
    else:
        st.info("Model artefact not found. Train the model via `python -m src.train` to enable predictions.")
        return

    st.write("Provide feature values to obtain a disposition prediction.")
    medians = df[list(KOI_FEATURE_COLUMNS)].median(numeric_only=True)
    with st.form("manual_prediction"):
        inputs: Dict[str, float] = {}
        for feature in KOI_FEATURE_COLUMNS:
            lower, upper = _infer_bounds(df, feature)
            default = float(medians.get(feature, (lower + upper) / 2))
            inputs[feature] = st.number_input(
                feature,
                value=default,
                min_value=float(lower),
                max_value=float(upper),
                step=(upper - lower) / 200 if upper > lower else 0.01,
                format="%.4f",
            )
        submitted = st.form_submit_button("Predict disposition")

    if submitted:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        st.success(f"Predicted disposition: **{prediction}**")
        prob_df = pd.DataFrame({"disposition": model.classes_, "probability": probabilities})
        prob_chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x="probability:Q",
                y=alt.Y("disposition:N", sort="-x"),
                tooltip=["disposition", alt.Tooltip("probability:Q", format=".2f")],
            )
        )
        st.altair_chart(prob_chart, use_container_width=True)


def render_batch_prediction(df: pd.DataFrame) -> None:
    st.subheader("Batch predictions")
    model = None
    if MODEL_PATH.exists():
        model = get_model()
    else:
        st.info("Model artefact not found. Train the model via `python -m src.train` to enable predictions.")
        return

    uploaded = st.file_uploader("Upload a CSV file with feature columns", type="csv")
    if uploaded is None:
        st.write("Your CSV must contain the following columns:")
        st.code(", ".join(KOI_FEATURE_COLUMNS))
        return

    try:
        user_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read CSV: {exc}")
        return

    missing = set(KOI_FEATURE_COLUMNS) - set(user_df.columns)
    if missing:
        st.error(f"Uploaded file is missing required columns: {sorted(missing)}")
        return

    feature_frame = user_df[list(KOI_FEATURE_COLUMNS)]
    predictions = model.predict(feature_frame)
    probabilities = model.predict_proba(feature_frame)
    result_df = user_df.copy()
    result_df["predicted_disposition"] = predictions
    for idx, disposition in enumerate(model.classes_):
        result_df[f"prob_{disposition}"] = probabilities[:, idx]

    st.download_button(
        "Download predictions",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="exoplanet_predictions.csv",
        mime="text/csv",
    )
    st.dataframe(result_df.head(100))


def main() -> None:
    dataset = get_dataset()
    metrics = get_metrics()

    render_metrics_section(metrics)
    render_dataset_section(dataset)

    st.header("Predict new candidates")
    tabs = st.tabs(["Manual input", "Batch predictions"])
    with tabs[0]:
        render_manual_prediction(dataset)
    with tabs[1]:
        render_batch_prediction(dataset)


if __name__ == "__main__":
    main()
