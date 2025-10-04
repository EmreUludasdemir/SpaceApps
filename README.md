# Exoplanet Transit Classification Toolkit

This project provides a reproducible machine-learning pipeline and interactive Streamlit dashboard for classifying candidates from NASA's Kepler Objects of Interest (KOI) catalogue. It automatically downloads publicly available KOI data, evaluates multiple ensemble models with cross-validation, and exposes an interface for exploring the dataset and predicting dispositions for new observations.

## Features

- Automated download of a curated subset of the KOI dataset directly from the NASA Exoplanet Archive `nstedAPI` service.
- Registry of ensemble classifiers (Gradient Boosting, Random Forest, Extra Trees, AdaBoost, Random Subspace, Stacking) with cross-validated scoring, auto-selection, and macro-metric tracking inspired by recent exoplanet vetting studies. [MNRAS 513, 5505](https://academic.oup.com/mnras/article/513/4/5505/6472249) discusses the importance of minimising false positives, while [Luz et al. 2024](https://www.mdpi.com/2079-9292/13/19/3950) benchmarks ensemble pipelines for KOI data.
- Persisted model artefacts (trained model, metrics, feature importances) for reproducible inference.
- Streamlit application with:
  - Performance dashboard (macro accuracy/F1/specificity, classification report, specificity-by-class, confusion matrix, cross-validation leaderboard, feature importance).
  - Dataset explorer with filtering and feature distribution visualisations.
  - Manual prediction form and batch CSV upload for new candidate classification.

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train the model** (downloads the latest KOI data and saves artefacts under `models/`):

   ```bash
   python -m src.train --auto-select
   ```

   Useful flags:

   - `--refresh-data` — force re-download of the KOI dataset.
   - `--model {name}` — train a specific ensemble from the registry.
   - `--auto-select` — evaluate all registered ensembles and persist the best performer.
   - `--selection-metric {metric}` — choose the metric used for automatic selection (default: `f1_macro`).
   - `--cv-splits N` — adjust the number of stratified folds used during cross-validation (default: `5`).

3. **Launch the Streamlit interface**:

   ```bash
   streamlit run app/streamlit_app.py
   ```

   The dashboard will be available at <http://localhost:8501> by default.

### Interpreting evaluation outputs

- **Cross-validation metrics** – When you run `python -m src.train`, the script reports macro accuracy, macro F1, and macro specificity that are computed from stratified k-fold cross-validation on the downloaded KOI dataset. These values reflect real training runs; re-running the command on your machine reproduces them with the same preprocessing and model definitions.
- **Artefact files** – The metrics displayed in the Streamlit dashboard are loaded from `models/metrics.json`, which is generated at the end of each training execution along with the saved estimator in `models/exoplanet_classifier.joblib` and feature importances in `models/feature_importances.json`.
- **Testing status in docs** – Some documentation snippets in this repository include a “⚠️ Tests not run (read-only review)” note. This simply indicates that, for that specific write-up, automated tests were not re-executed; it does not invalidate the cross-validation metrics produced by the training pipeline.

## Research alignment

The expanded evaluation workflow follows guidance from recent literature that emphasises ensemble diversity, cross-validation, and explicit monitoring of false-positive rates when working with KOI- and TESS-like catalogues. [Luz et al. 2024](https://www.mdpi.com/2079-9292/13/19/3950) demonstrate that tuned ensemble methods (e.g., Random Forest, Extra Trees, Stacking) deliver superior macro metrics across KOI folds; we mirror this by exposing a registry of comparable ensembles, exporting fold-wise timings, and ranking models by macro F1. Complementary recommendations from [MNRAS 513, 5505](https://academic.oup.com/mnras/article/513/4/5505/6472249) motivate tracking specificity alongside precision/recall to better differentiate astrophysical false positives, so the dashboard now surfaces macro specificity and class-wise true negative rates.

## Repository layout

```
.
├── app/                   # Streamlit front-end
│   └── streamlit_app.py
├── data/                  # Cached KOI dataset (downloaded automatically)
├── models/                # Trained model and metrics outputs
├── requirements.txt       # Python dependencies
└── src/
    ├── __init__.py
    ├── data.py            # Data download and loading helpers
    ├── model.py           # Model construction and evaluation utilities
    └── train.py           # Training entry point with cross-validation
```

## Data source

The project queries the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) KOI catalogue via the legacy [`nstedAPI` endpoint](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#nstead) to retrieve a compact subset of informative features alongside the `koi_disposition` label. Downloads remain lightweight while preserving the key science context.

## Extending the project

- Add or adjust ensemble definitions in `src/model.py` to experiment with alternative hyperparameters or algorithms.
- Incorporate additional features from the KOI table or other missions (K2, TESS) by updating `src/data.py`.
- Log training runs with MLflow or Weights & Biases for experiment tracking.
- Deploy the Streamlit app (e.g., Streamlit Cloud, Hugging Face Spaces) for easy sharing with collaborators.
