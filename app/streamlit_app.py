"""Streamlit dashboard for the optimised exoplanet classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import TARGET_COLUMN, load_koi_dataframe, preprocess_koi_dataframe
from src.model import load_metrics, load_model

MODEL_PATH = PROJECT_ROOT / "models" / "exoplanet_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Exoplanet Transit Classifier", layout="wide")
st.title("Exoplanet Transit Classification Toolkit")


@st.cache_resource(show_spinner=True)
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=True)
def get_dataset_payload() -> Dict[str, object]:
    prepared = preprocess_koi_dataframe(load_koi_dataframe(), return_identifiers=True)
    inverse_mapping = {v: k for k, v in prepared.label_mapping.items()}
    frame = prepared.features.copy()
    if prepared.identifiers is not None:
        frame = pd.concat([prepared.identifiers.reset_index(drop=True), frame], axis=1)
    frame[TARGET_COLUMN] = prepared.target.map(inverse_mapping)
    return {
        "frame": frame,
        "numeric_features": prepared.numeric_features,
        "categorical_features": prepared.categorical_features,
        "label_mapping": prepared.label_mapping,
        "identifier_columns": list(prepared.identifiers.columns) if prepared.identifiers is not None else [],
    }


@st.cache_data(show_spinner=False)
def get_metrics() -> Dict[str, object] | None:
    if METRICS_PATH.exists():
        return load_metrics(METRICS_PATH)
    return None


def _extract_summary_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    if not metrics:
        return {}
    payload = metrics.get("metrics") if isinstance(metrics, dict) else None
    if isinstance(payload, dict):
        return payload
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro"]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _build_cv_summary_table(metrics: Dict[str, object]) -> pd.DataFrame:
    cv_section = metrics.get("cross_validation", {}) if metrics else {}
    summary = cv_section.get("summary", {}) if isinstance(cv_section, dict) else {}
    rows = []
    for metric, stats in summary.items():
        if not isinstance(stats, dict):
            continue
        rows.append({"Metric": metric, "Mean": stats.get("mean"), "Std": stats.get("std")})
    return pd.DataFrame(rows)


def _infer_bounds(df: pd.DataFrame, feature: str) -> tuple[float, float]:
    series = df[feature].dropna()
    if series.empty:
        return (0.0, 1.0)
    lower = float(series.quantile(0.01))
    upper = float(series.quantile(0.99))
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def render_metrics_section(metrics: Dict[str, object] | None) -> None:
    st.header("Model performance")
    if not metrics:
        st.warning("No metrics found. Run `python -m src.train` to train the model and populate metrics.")
        return

    summary = _extract_summary_metrics(metrics)
    cols = st.columns(3)
    cols[0].metric("Validation accuracy", f"{summary.get('accuracy', float('nan')):.3f}")
    cols[1].metric("Macro F1", f"{summary.get('f1_macro', float('nan')):.3f}")
    cols[2].metric("Macro specificity", f"{summary.get('specificity_macro', float('nan')):.3f}")

    validation = metrics.get("validation", {}) if isinstance(metrics, dict) else {}
    if validation:
        threshold = validation.get("best_threshold", 0.5)
        baseline_threshold = validation.get("baseline_threshold", 0.5)
        st.markdown(
            f"**Optimised decision threshold:** `{threshold:.3f}` (baseline `{baseline_threshold:.2f}`)"
        )
        baseline_metrics = validation.get("baseline_metrics", {})
        best_metrics = validation.get("best_metrics", {})
        comparison = pd.DataFrame({"Baseline": baseline_metrics, "Optimised": best_metrics})
        st.dataframe(comparison, use_container_width=True)

        specificity_by_class = validation.get("specificity_by_class", {})
        if specificity_by_class:
            class_specificity = {k: v for k, v in specificity_by_class.items() if k != "macro"}
            spec_df = (
                pd.Series(class_specificity)
                .rename("Specificity")
                .to_frame()
                .reset_index()
                .rename(columns={"index": "Disposition"})
            )
            spec_chart = (
                alt.Chart(spec_df)
                .mark_bar()
                .encode(x="Specificity:Q", y=alt.Y("Disposition:N", sort="-x"))
                .properties(title="Specificity by class", width=400, height=220)
            )
            st.altair_chart(spec_chart, use_container_width=False)

        confusion = np.array(validation.get("confusion_matrix") or [])
        if confusion.size:
            mapping = metrics.get("label_mapping", {})
            labels = sorted(mapping, key=mapping.get) if isinstance(mapping, dict) else list(mapping)
            conf_df = pd.DataFrame(confusion, index=labels, columns=labels)
            heatmap = (
                alt.Chart(conf_df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count"))
                .mark_rect()
                .encode(x="Predicted:O", y="index:O", color="Count:Q", tooltip=["index", "Predicted", "Count"])
                .properties(width=320, height=320, title="Validation confusion matrix")
            )
            st.altair_chart(heatmap, use_container_width=False)

        report = validation.get("classification_report")
        if report:
            class_report_df = pd.DataFrame(report).T
            rename_map = {"precision": "Precision", "recall": "Recall", "f1-score": "F1-score"}
            class_report_df = class_report_df.rename(columns=rename_map)
            st.dataframe(
                class_report_df.style.format(
                    {col: "{:.2f}" for col in ["Precision", "Recall", "F1-score"] if col in class_report_df}
                ),
                use_container_width=True,
            )

    threshold_section = metrics.get("threshold", {}) if isinstance(metrics, dict) else {}
    if threshold_section:
        candidates = threshold_section.get("candidates", [])
        if candidates:
            threshold_df = pd.DataFrame(candidates)
            tooltips = [alt.Tooltip("threshold:Q", format=".3f")]
            for column in threshold_df.columns:
                if column == "threshold":
                    continue
                tooltips.append(alt.Tooltip(f"{column}:Q", format=".3f"))
            chart = (
                alt.Chart(threshold_df)
                .mark_line()
                .encode(
                    x="threshold:Q",
                    y="specificity_macro:Q",
                    tooltip=tooltips,
                )
                .properties(title="Threshold sweep", height=260)
            )
            st.altair_chart(chart, use_container_width=True)

    full_dataset = metrics.get("full_dataset", {}) if isinstance(metrics, dict) else {}
    if full_dataset:
        st.subheader("Full KOI dataset evaluation")
        full_metrics = full_dataset.get("metrics", {})
        if full_metrics:
            full_df = pd.DataFrame(full_metrics, index=["Value"]).T
            st.dataframe(full_df.style.format("{:.3f}"), use_container_width=True)
        confusion = np.array(full_dataset.get("confusion_matrix") or [])
        if confusion.size:
            mapping = metrics.get("label_mapping", {})
            labels = sorted(mapping, key=mapping.get) if isinstance(mapping, dict) else list(mapping)
            conf_df = pd.DataFrame(confusion, index=labels, columns=labels)
            heatmap = (
                alt.Chart(conf_df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count"))
                .mark_rect()
                .encode(x="Predicted:O", y="index:O", color="Count:Q", tooltip=["index", "Predicted", "Count"])
                .properties(width=320, height=320, title="Full dataset confusion matrix")
            )
            st.altair_chart(heatmap, use_container_width=False)

    cv_table = _build_cv_summary_table(metrics)
    if not cv_table.empty:
        st.subheader("Cross-validation summary")
        st.dataframe(cv_table.style.format({"Mean": "{:.3f}", "Std": "{:.3f}"}), use_container_width=True)

    tuning = metrics.get("tuning") if isinstance(metrics, dict) else None
    if tuning and tuning.get("best_params"):
        st.subheader("Tuned hyperparameters")
        st.json(tuning.get("best_params"))

    top_predictions = metrics.get("top_predictions", {}) if isinstance(metrics, dict) else {}
    records = top_predictions.get("records", []) if isinstance(top_predictions, dict) else []
    if records:
        st.subheader("Highest-confidence candidate predictions")
        df = pd.json_normalize(records)
        display_cols = [col for col in df.columns if not col.startswith("features.")]
        feature_cols = sorted([col for col in df.columns if col.startswith("features.")])
        ordered_cols = display_cols + feature_cols
        st.dataframe(df[ordered_cols], use_container_width=True)


def render_dataset_section(dataset_payload: Dict[str, object]) -> None:
    st.header("Explore the Kepler Objects of Interest dataset")
    frame: pd.DataFrame = dataset_payload["frame"]
    with st.expander("Dataset preview", expanded=True):
        dispositions = sorted(frame[TARGET_COLUMN].unique())
        disposition_filter = st.multiselect(
            "Filter by disposition", dispositions, default=list(dispositions)
        )
        filtered = frame[frame[TARGET_COLUMN].isin(disposition_filter)]
        st.write(f"Showing {len(filtered):,} of {len(frame):,} curated KOI records.")
        st.dataframe(filtered.head(100))

    with st.expander("Feature distribution", expanded=False):
        feature = st.selectbox(
            "Select feature", dataset_payload["numeric_features"], key="feature_distribution"
        )
        chart = (
            alt.Chart(frame)
            .mark_bar(opacity=0.75)
            .encode(
                x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=40)),
                y="count()",
                color=f"{TARGET_COLUMN}:N",
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


def render_manual_prediction(dataset_payload: Dict[str, object]) -> None:
    st.subheader("Manual input")
    if not MODEL_PATH.exists():
        st.info("Model artefact not found. Train the model via `python -m src.train` to enable predictions.")
        return

    model = get_model()
    frame: pd.DataFrame = dataset_payload["frame"]
    numeric_features = list(dataset_payload["numeric_features"])
    categorical_features = list(dataset_payload["categorical_features"])
    medians = frame[numeric_features].median(numeric_only=True)

    with st.form("manual_prediction"):
        inputs: Dict[str, float | str] = {}
        for feature in numeric_features:
            lower, upper = _infer_bounds(frame, feature)
            default = float(medians.get(feature, (lower + upper) / 2))
            inputs[feature] = st.number_input(
                feature,
                value=default,
                min_value=float(lower),
                max_value=float(upper),
                step=(upper - lower) / 200 if upper > lower else 0.01,
                format="%.4f",
            )
        for feature in categorical_features:
            options = sorted(frame[feature].fillna("Unknown").unique())
            default = options[0] if options else "Unknown"
            inputs[feature] = st.selectbox(feature, options, index=options.index(default) if default in options else 0)
        submitted = st.form_submit_button("Predict disposition")

    if submitted:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        st.success(f"Predicted disposition: **{prediction}**")
        prob_df = pd.DataFrame({"disposition": model.classes_, "probability": probabilities})
        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(x="probability:Q", y=alt.Y("disposition:N", sort="-x"))
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)


def render_batch_prediction(dataset_payload: Dict[str, object]) -> None:
    st.subheader("Batch predictions")
    if not MODEL_PATH.exists():
        st.info("Model artefact not found. Train the model via `python -m src.train` to enable predictions.")
        return

    model = get_model()
    numeric_features = list(dataset_payload["numeric_features"])
    categorical_features = list(dataset_payload["categorical_features"])
    expected_columns = numeric_features + categorical_features

    uploaded = st.file_uploader("Upload a CSV file with feature columns", type="csv")
    if uploaded is None:
        st.write("Your CSV must contain the following columns:")
        st.code(", ".join(expected_columns))
        return

    try:
        user_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read CSV: {exc}")
        return

    missing = set(expected_columns) - set(user_df.columns)
    if missing:
        st.error(f"Uploaded file is missing required columns: {sorted(missing)}")
        return

    feature_frame = user_df[expected_columns]
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
    dataset_payload = get_dataset_payload()
    metrics = get_metrics()

    render_metrics_section(metrics)
    render_dataset_section(dataset_payload)

    st.header("Predict new candidates")
    tabs = st.tabs(["Manual input", "Batch predictions"])
    with tabs[0]:
        render_manual_prediction(dataset_payload)
    with tabs[1]:
        render_batch_prediction(dataset_payload)


if __name__ == "__main__":
    main()
