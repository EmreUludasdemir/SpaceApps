# Exoplanet Transit Classification Toolkit

This project provides a reproducible machine-learning pipeline and interactive Streamlit dashboard for classifying candidates from NASA's Kepler Objects of Interest (KOI) catalogue. It automatically downloads the public KOI table, performs the preprocessing steps recommended in the ensemble-focused literature, and trains an optimised stacking model that maximises macro specificity while maintaining strong macro F1 and accuracy. The resulting artefacts feed a Streamlit interface for exploring the balanced dataset, inspecting evaluation outputs, tuning decision thresholds, and predicting dispositions for new observations.

## Features

- Automated KOI ingestion: the training script retrieves the latest catalogue via NASA's `nstedAPI`, removes identifier columns, filters to confirmed/candidate dispositions, converts the target into a binary label (`CONFIRMED` = 0, `CANDIDATE` = 1), imputes categorical delivery names, and balances the classes through down-sampling.
- Stacking ensemble tailored to the KOI task: Random Forest, Extra Trees, and XGBoost base learners feed a logistic regression meta-learner (as recommended by [MNRAS 513, 5505](https://academic.oup.com/mnras/article/513/4/5505/6472249) and [Luz et al. 2024](https://www.mdpi.com/2079-9292/13/19/3950)).
- Hyperparameter optimisation: `GridSearchCV` evaluates a curated grid under 10-fold, 5-repeat stratified cross-validation and selects the configuration with the highest macro specificity (breaking ties with macro F1).
- Threshold tuning: after training, the script sweeps probability thresholds on a validation split, optimising macro specificity while enforcing the baseline macro F1, and records the best trade-off alongside a precision–recall curve.
- Persisted artefacts: the tuned stacking pipeline, threshold wrapper, metrics (including per-fold diagnostics, validation confusion matrix, and threshold sweep), and feature lists are saved under `models/`.
- Streamlit dashboard with:
  - Validation metrics (accuracy, macro F1, macro specificity), class-wise specificity bars, confusion matrix, classification report, threshold sweep visualisation, and cross-validation summary.
  - Balanced dataset explorer with filtering and feature distribution charts.
  - Manual prediction form and batch CSV upload that respect the tuned threshold for inference.

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train and optimise the model** (downloads the latest KOI data and saves artefacts under `models/`):

   ```bash
   python -m src.train
   ```

   By default the command performs 10×5 repeated stratified cross-validation, grid-searches the stacking ensemble hyperparameters, tunes the probability threshold on a held-out validation split, and finally retrains the model on the balanced dataset with the best configuration. Training can take several minutes on a CPU-only machine because of the repeated CV and GridSearchCV runs.

   Useful flags:

   - `--refresh-data` — force a fresh download of the KOI dataset.
   - `--validation-size 0.2` — change the proportion of balanced data reserved for threshold optimisation.
   - `--cv-folds 10` / `--cv-repeats 5` — adjust the repeated stratified CV strategy.
   - `--disable-tuning` — skip the GridSearchCV stage and use the baseline stacking parameters (helpful for quick iteration).
   - `--model-path`, `--metrics-path` — customise where the trained artefacts are written.

3. **Launch the Streamlit interface**:

   ```bash
   streamlit run app/streamlit_app.py
   ```

   The dashboard will be available at <http://localhost:8501> by default.

### Interpreting evaluation outputs

- **Cross-validation metrics** – The `cross_validation` section of `models/metrics.json` stores the mean and standard deviation of accuracy, macro precision/recall/F1, and macro specificity across all 50 folds, as well as fold-level true negative/false positive counts.
- **Validation metrics & threshold** – The `validation` section captures the baseline (0.5) threshold results and the optimised threshold metrics, confusion matrix, per-class specificity, and classification report. These are the values surfaced in the Streamlit dashboard.
- **Threshold sweep** – `threshold.candidates` records the macro specificity and macro F1 achieved at each evaluated probability threshold so you can audit the precision–recall trade-off. The chosen threshold is also embedded in the persisted model and used for all manual/batch predictions.
- **Artefact files** – Training produces `models/exoplanet_classifier.joblib` (stacking pipeline wrapped with the tuned threshold) and `models/metrics.json` (detailed evaluation payload). Re-running the command with the same random seed reproduces the metrics thanks to deterministic preprocessing and balancing.

## Research alignment

The training pipeline mirrors the best practices highlighted by recent ensemble studies on KOI data. [MNRAS 513, 5505](https://academic.oup.com/mnras/article/513/4/5505/6472249) emphasises the need for ensemble diversity, repeated cross-validation, and explicit monitoring of specificity to keep false positives under control. [Luz et al. 2024](https://www.mdpi.com/2079-9292/13/19/3950) reports that stacking combinations of tree-based models yield superior macro metrics once hyperparameters are tuned and imbalanced classes are addressed. This repository encodes those recommendations via the balanced stacking pipeline, repeated CV grid search, macro specificity-driven selection, and post-hoc threshold optimisation exposed through the dashboard.

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
    ├── data.py            # Data download, preprocessing, and balancing helpers
    ├── model.py           # Stacking pipeline, CV, threshold optimisation utilities
    └── train.py           # Training entry point with grid search + validation workflow
```

## Data source

The project queries the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) KOI catalogue via the legacy [`nstedAPI` endpoint](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#nstead) to retrieve a compact subset of informative features alongside the `koi_disposition` label. Downloads remain lightweight while preserving the key science context.

## Extending the project

- Add or adjust ensemble definitions in `src/model.py` to experiment with alternative hyperparameters or algorithms.
- Incorporate additional features from the KOI table or other missions (K2, TESS) by updating `src/data.py`.
- Log training runs with MLflow or Weights & Biases for experiment tracking.
- Deploy the Streamlit app (e.g., Streamlit Cloud, Hugging Face Spaces) for easy sharing with collaborators.
