"""Data loading and preprocessing utilities for the NASA Kepler KOI dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "koi_disposition"
IDENTIFIER_COLUMNS: Sequence[str] = (
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_pdisposition",
    "koi_score",
)

NUMERIC_FEATURE_COLUMNS: Sequence[str] = (
    "koi_period",
    "koi_time0bk",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_slogg",
    "koi_srad",
    "koi_steff",
    "koi_kepmag",
)
"""Continuous predictors retained after removing identifiers."""

CATEGORICAL_FEATURE_COLUMNS: Sequence[str] = ("koi_tce_delivname",)
"""Categorical predictors that require dummy encoding."""

KOI_FEATURE_COLUMNS: Sequence[str] = (*NUMERIC_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS)

NSTED_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = MODULE_ROOT / "data" / "kepler_koi.csv"

# The `select=*` clause ensures the archive returns all available columns so the
# preprocessing stage can drop identifiers and construct the engineered view
# described in the research prompt.
_SELECT_COLUMNS = "*"


@dataclass(frozen=True)
class PreparedDataset:
    """Container holding features, target labels, and associated metadata."""

    features: pd.DataFrame
    target: pd.Series
    numeric_features: Sequence[str]
    categorical_features: Sequence[str]
    label_mapping: dict[str, int]


def _build_request_params() -> dict[str, str]:
    return {
        "table": "koi",
        "select": _SELECT_COLUMNS,
        "where": "koi_disposition is not null",
        "format": "csv",
    }


def download_koi_dataset(destination: Path | None = None, *, overwrite: bool = False) -> Path:
    """Download the KOI dataset from the NASA Exoplanet Archive."""

    destination = Path(destination or DEFAULT_CACHE_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        LOGGER.info("Dataset already cached at %s", destination)
        return destination

    params = _build_request_params()
    LOGGER.info("Downloading KOI dataset from %s", NSTED_ENDPOINT)
    response = requests.get(NSTED_ENDPOINT, params=params, timeout=120)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")
    LOGGER.info("Saved KOI dataset to %s (%.2f MB)", destination, destination.stat().st_size / 1e6)
    return destination


def load_koi_dataframe(cache_path: Path | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Load the KOI dataset from the local cache, downloading it if necessary."""

    cache_path = download_koi_dataset(cache_path, overwrite=refresh)
    df = pd.read_csv(cache_path, low_memory=False)
    LOGGER.debug("Loaded KOI dataframe with shape %s", df.shape)
    return df


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def preprocess_koi_dataframe(df: pd.DataFrame) -> PreparedDataset:
    """Clean the KOI dataframe and return a balanced training view."""

    df = df.copy()
    df.drop(columns=[col for col in IDENTIFIER_COLUMNS if col in df.columns], inplace=True, errors="ignore")

    # Keep only confirmed and candidate dispositions.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.upper()
    valid_labels = {"CONFIRMED": 0, "CANDIDATE": 1}
    df = df[df[TARGET_COLUMN].isin(valid_labels)].copy()
    df.sort_values("koi_period", inplace=True)

    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = "Unknown"
        df[column] = df[column].fillna("Unknown").astype(str)

    _ensure_columns(df, NUMERIC_FEATURE_COLUMNS)

    for column in NUMERIC_FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURE_COLUMNS)

    df["target"] = df[TARGET_COLUMN].map(valid_labels)
    df = df.dropna(subset=["target"])  # defensive: should be satisfied already

    features = df[list(KOI_FEATURE_COLUMNS)].reset_index(drop=True)
    target = df["target"].astype(int).reset_index(drop=True)

    return PreparedDataset(
        features=features,
        target=target,
        numeric_features=list(NUMERIC_FEATURE_COLUMNS),
        categorical_features=[col for col in CATEGORICAL_FEATURE_COLUMNS if col in features.columns],
        label_mapping=valid_labels,
    )


__all__ = [
    "CATEGORICAL_FEATURE_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "KOI_FEATURE_COLUMNS",
    "NUMERIC_FEATURE_COLUMNS",
    "PreparedDataset",
    "TARGET_COLUMN",
    "download_koi_dataset",
    "load_koi_dataframe",
    "preprocess_koi_dataframe",
]
