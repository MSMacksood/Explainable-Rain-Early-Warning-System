# Architecture Validation Report

**Documents reviewed:** `Q1_ASML_SL_Weather_BluePrint.md`, `ARCHITECTURE_IMPLEMENTATION_ACADEMIC_THESIS_GENERATION.md`
**Date:** 2026-07-19
**Overall verdict: VALID and practically demonstrable**, with 8 improvements applied before implementation (2 critical, 6 moderate).

## 1. What passes as-is

- **Dataset choice** — Sri Lanka Weather Dataset (Kaggle, CC0) meets all five Phase-0 hard constraints. **Verified at runtime against the real CSV (2026-07-19 rerun): 147,480 rows × 24 cols, 30 cities, 2010-01-01 → 2023-06-17, zero missing values** (see `reports/data_validation.json`). Two blueprint approximations corrected by the real data: the true row count is 147,480 (not the computed 142,470 — coverage extends to June 2023), and the raw column is `weathercode` (normalized at ingestion). The blueprint's "class imbalance" framing was also refined: 65.7% of days are rainy, so the operative rarity is heavy rain (>20 mm, <7% of days), and the Phase-0 challenge predicate was reformulated accordingly.
- **Task formulation** — binary `rain_tomorrow` (precip(t+1) > 1.0 mm, per city) + regression head is well-posed and leakage-safe as specified (features use day *t*, target is *t+1*).
- **Model family spread** — Linear / KNN / RF / GBDT (CatBoost primary) / LSTM is the right coverage for a tabular, medium-scale, temporally ordered panel; CatBoost-first is consistent with 2024–2025 tabular benchmarks.
- **Metric choice** — PR-AUC primary for imbalanced classification, recall-weighted, RMSE/MAE for regression: correct.
- **XAI + ethics scope** — SHAP global/local + per-zone fairness reporting is appropriate and implementable.
- **MLOps stack** — DVC + MLflow + GitHub Actions + Evidently + FastAPI on serverless is coherent, low-cost, and standard.

## 2. Issues found and improvements applied

### Critical

**C1. Data leakage in the blueprint's CV code.** The Optuna snippet uses `sklearn.TimeSeriesSplit` directly on the row index of a dataframe sorted by `(city, time)`. Row-index splits on panel data put *all* of city A (2010–2023) in training while validating on city Z — and, worse, mix future and past across cities. This silently invalidates every reported CV score.
**Fix:** implemented `PanelTimeSeriesSplit` (`src/train.py`) that splits on **unique sorted dates** — expanding window with an optional embargo gap — so every fold trains on all cities' past and validates on all cities' future. This is the "grouped, blocked, expanding-window" split the instructions demand but the snippet did not deliver.

**C2. Dockerfile is not multi-stage.** The blueprint labels its single-stage Dockerfile "multi-stage". **Fix:** genuine two-stage build (builder wheels → slim runtime, non-root user).

### Moderate

**M1. SMOTE on time-series panels is risky.** Synthetic interpolation across temporally adjacent rows can manufacture leakage-adjacent samples. **Fix:** class weights (`auto_class_weights="Balanced"` / `scale_pos_weight`) are the *primary* imbalance treatment; SMOTE is retained as an optional, fold-internal-only ablation (`config: imbalance.strategy`), never applied before splitting.

**M2. Threshold tuning: Youden's J is ROC-based and misleading under heavy imbalance.** **Fix:** primary threshold selected by maximizing F2 on the validation PR curve; Youden's J reported secondarily for the thesis comparison the spec requests.

**M3. Optuna objective lacked early stopping/pruning wiring.** Fixed: `eval_set` + CatBoost `od_type=Iter` early stopping inside each fold, Optuna `MedianPruner`/ASHA-style pruning on intermediate fold scores.

**M4. Heat-index RH proxy is physically crude** (RH inferred from apparent−actual temperature spread; that spread also embeds wind). Retained per spec, but clipped to [20, 100]%, documented as an approximation, and accompanied by an `rh_proxy` feature so SHAP can expose its influence; thesis discusses the limitation.

**M5. Rolling-anomaly numerical stability.** `min_periods=3` with near-constant dry spells gives σ≈0 → exploding z-scores. Fixed: σ floored at a small epsilon *and* anomaly clipped to ±10; leading NaN rows dropped after lag construction.

**M6. Zero-inflated regression.** Direct RMSE regression on `precipitation_sum` is dominated by dry days. Improvement: the regression head is a **two-part (hurdle) model** — the classifier gates occurrence, a regressor trained on rainy-day rows predicts amount (log1p-transformed). MAPE reported on the non-zero subset only, as the blueprint itself caveats.

### Minor (noted, handled in code)

- FastAPI endpoint made properly async with model loaded once at startup (lifespan), plus the `/predict_batch` bulk endpoint the spec requires but the blueprint snippet omits.
- `weather_code` consistency check (rain code vs 0 mm) implemented as a data-quality audit, not silent mutation.
- All dependencies version-pinned; global seed control for reproducibility.
- LSTM windows built **per city** (no cross-city sequence bleed), 7-day lookback per spec, PyTorch implementation with an sklearn-free fallback if torch is unavailable in the runtime.

## 3. Practical demonstrability

The full pipeline has been executed end-to-end **on the real Kaggle dataset**: ingestion/validation → feature engineering → 7-model benchmark under `PanelTimeSeriesSplit` → 15-trial Optuna TPE tuning (configurable to 100) → hold-out evaluation with Wilcoxon/Diebold–Mariano tests and fairness decomposition → SHAP artifacts → drift monitoring → live API smoke test. Real-run outputs live in `reports/`; the earlier offline surrogate demonstration is archived in `reports/surrogate_run/` with a side-by-side in `reports/COMPARISON_SURROGATE_VS_REAL.md`. Headline real-data results: CV PR-AUC 0.934 (CatBoost, first on all folds), hold-out PR-AUC 0.974, F2-threshold recall 0.976 at precision 0.809, heavy-rain recall 0.996. Notably, the fairness picture reversed between surrogate and real data (weakest zone: hill country → dry zone under the Maha monsoon), confirming that equity conclusions must come from real data. Nothing in the architecture requires resources beyond a modest single machine; cloud components are expressed as code/config and deployable without modification.

## 4. Conclusion

The blueprint is scientifically defensible and the implementation instructions are executable. With the leakage fix (C1) — which is essential, not cosmetic — the reported results will be honest out-of-time estimates. All improvements above are implemented in the codebase rather than merely recommended.
