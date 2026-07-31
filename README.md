# Monsoon-Aware, Explainable Rain Early-Warning System for Sri Lanka

Production-grade implementation of the *Applied Supervised ML Blueprint* — a 30-city,
next-day rain classification + precipitation regression system with SHAP explainability,
fairness auditing, and a full MLOps loop (DVC · MLflow · Optuna · Evidently · FastAPI · Docker · GitHub Actions).

## Repository layout

```text
├── .github/workflows/ci-cd.yaml   # lint → test → build → smoke → canary deploy → drift check
├── config/config.yaml             # single source of configuration
├── data/                          # raw + processed (DVC-managed, gitignored)
├── models/                        # serialized CatBoost artifacts + threshold
├── reports/                       # validation, training, evaluation, drift, SHAP artifacts
├── scripts/
│   ├── make_demo_data.py          # SURROGATE data generator (offline demo only)
│   └── run_step.py                # chunked/resumable pipeline driver
├── src/
│   ├── data_prep.py               # ingestion, Phase-0 validation, cleaning
│   ├── features.py                # 6 engineered features, lags, targets, selection
│   ├── train.py                   # PanelTimeSeriesSplit, 5 families + MLP + LSTM, Optuna
│   ├── evaluate.py                # metrics, thresholds, Wilcoxon/DM tests, fairness
│   ├── explain.py                 # SHAP global/local + permutation importance
│   └── utils.py
├── app/
│   ├── main.py                    # async FastAPI: /health /predict /predict_batch
│   ├── schemas.py                 # Pydantic v2 request/response models
│   └── monitoring.py              # Evidently + scipy KS/Chi-square drift monitor
├── tests/test_pipeline.py         # leakage-safety, feature math, schema tests
├── Dockerfile                     # genuine multi-stage build, non-root runtime
├── dvc.yaml                       # 5-stage reproducible pipeline
├── mlflow_setup.sh                # local file-store or SQLite-backed server
└── requirements.txt               # pinned dependencies
```

## Quick start

```bash
pip install -r requirements.txt

# Real data: place the Kaggle CSV at data/raw/SriLanka_Weather_Dataset.csv
#   (https://www.kaggle.com/datasets/rasulmah/sri-lanka-weather-dataset)
#   — or let src.data_prep download it via kagglehub.
# Offline demo: python scripts/make_demo_data.py

dvc repro                 # or run stages manually:
python -m src.data_prep && python -m src.features
python -m src.train       # full benchmark + Optuna tuning
python -m src.evaluate && python -m src.explain
python -m app.monitoring  # drift check (exit 1 on alert)

uvicorn app.main:app --port 8080   # serve
pytest tests -q                    # tests
```

Constrained environments (CI matrices, hard-timeout shells): `scripts/run_step.py`
executes one (family, fold) or a chunk of Optuna trials per invocation and
checkpoints everything, producing identical artifacts.

## Results (REAL Kaggle dataset, 147,480 rows, 2010–2023; 15-trial Optuna budget)

| Model | PR-AUC (5-fold blocked CV) |
|---|---|
| CatBoost (default / tuned) | **0.9337 ± 0.0087 / 0.9335** |
| Logistic Regression | 0.9322 ± 0.0095 |
| Random Forest | 0.9313 ± 0.0089 |
| XGBoost | 0.9296 ± 0.0079 |
| LightGBM | 0.9283 ± 0.0089 |
| MLP (deep family, tabular) | 0.9078 ± 0.0118 |
| KNN | 0.9076 ± 0.0103 |

Hold-out (final 15% of dates, Jun 2021–Jun 2023): PR-AUC 0.974, ROC-AUC 0.940;
F2-tuned threshold (0.136) gives recall 0.976 / precision 0.809 / F1 0.885;
heavy-rain (>20 mm) recall 0.996. Hurdle regression: RMSE 6.45 mm, MAE 3.10 mm, R² 0.46.
Per-zone PR-AUC: wet 0.988, hill 0.942, dry 0.886 — the dry-zone/Maha-monsoon gap drives
the fairness audit in the thesis. DM vs climatology: p < 0.001.

Surrogate-run demo artifacts are archived in `reports/surrogate_run/`; see
`reports/COMPARISON_SURROGATE_VS_REAL.md` for the full side-by-side.

## Key engineering decisions

See `VALIDATION_REPORT.md` — most importantly `PanelTimeSeriesSplit`, which replaces
the blueprint's row-index `TimeSeriesSplit` (a silent leakage bug on panel data) with
an expanding-window split over unique dates with an embargo gap, unit-tested in
`tests/test_pipeline.py`.
