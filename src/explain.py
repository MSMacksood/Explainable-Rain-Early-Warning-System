"""Explainability: SHAP global/local artifacts + permutation importance.

Blueprint Phase 5(a): TreeExplainer summary (global), dependence
interactions, per-prediction waterfall (local), and permutation importance
as a robustness cross-check against correlated-feature artifacts.
LIME is optional (model-agnostic cross-check) and guarded by import.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)

REPORT_DIR = Path("reports/explainability")


def shap_global(clf, X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Global SHAP summary: beeswarm + mean-|SHAP| ranking CSV."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(clf)
    pool = Pool(X, cat_features=cat_cols)
    values = explainer.shap_values(pool)
    if isinstance(values, list):  # binary case can return [neg, pos]
        values = values[1]

    plt.figure()
    shap.summary_plot(values, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_beeswarm.png", dpi=150)
    plt.close("all")

    ranking = (
        pd.DataFrame({"feature": X.columns,
                      "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(REPORT_DIR / "shap_ranking.csv", index=False)

    # dependence plot for the top continuous driver
    top_cont = next(f for f in ranking["feature"] if f not in cat_cols)
    plt.figure()
    shap.dependence_plot(top_cont, values, X, show=False,
                         interaction_index="month_sin"
                         if "month_sin" in X.columns else None)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"shap_dependence_{top_cont}.png", dpi=150)
    plt.close("all")
    return ranking


def shap_local(clf, X: pd.DataFrame, cat_cols: list[str], indices: list[int]) -> None:
    """Local waterfall plots for individual predictions (case studies)."""
    explainer = shap.TreeExplainer(clf)
    for i in indices:
        row = X.iloc[[i]]
        sv = explainer.shap_values(Pool(row, cat_features=cat_cols))
        if isinstance(sv, list):
            sv = sv[1]
        exp = shap.Explanation(
            values=sv[0], base_values=explainer.expected_value,
            data=row.iloc[0].values, feature_names=list(X.columns))
        plt.figure()
        shap.plots.waterfall(exp, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"shap_waterfall_row{i}.png", dpi=150)
        plt.close("all")


def permutation_check(clf, X, y, cat_cols, seed: int, n_repeats: int = 3) -> pd.DataFrame:
    """Permutation importance on PR-AUC — robustness cross-check."""

    class _Wrapper:
        """Duck-typed estimator so sklearn can score a CatBoost model."""

        def __init__(self, model):
            self.model = model
            self._estimator_type = "classifier"
            self.classes_ = np.array([0, 1])

        def fit(self, *a, **k):
            return self

        def predict_proba(self, X_):
            return self.model.predict_proba(X_)

    def scorer(est, X_, y_):
        return average_precision_score(y_, est.predict_proba(X_)[:, 1])

    result = permutation_importance(
        _Wrapper(clf), X, y, scoring=scorer, n_repeats=n_repeats,
        random_state=seed)
    out = (
        pd.DataFrame({"feature": X.columns,
                      "importance_mean": result.importances_mean,
                      "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    out.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    return out


def run(config_path: str = "config/config.yaml", sample_n: int = 5000) -> None:
    cfg = load_config(config_path)
    df = pd.read_parquet(cfg["data"]["processed_parquet"])
    df["time"] = pd.to_datetime(df["time"])
    spec = joblib.load("models/feature_spec.joblib")
    feat_cols, cat_cols = spec["feat_cols"], spec["cat_cols"]

    clf = CatBoostClassifier()
    clf.load_model(cfg["serving"]["model_path"])

    # explain on the most recent slice (production-relevant regime)
    df = df.sort_values("time").tail(sample_n).reset_index(drop=True)
    X, y = df[feat_cols], df[cfg["target"]["classification"]]

    ranking = shap_global(clf, X, cat_cols)
    # local case studies: highest-prob rain day, a miss-risk day, a dry day
    proba = clf.predict_proba(X)[:, 1]
    cases = [int(np.argmax(proba)), int(np.argmin(np.abs(proba - 0.5))),
             int(np.argmin(proba))]
    shap_local(clf, X, cat_cols, cases)
    perm = permutation_check(clf, X, y, cat_cols, cfg["seed"])

    save_json(
        {"top10_shap": ranking.head(10).to_dict("records"),
         "top10_permutation": perm.head(10).to_dict("records"),
         "local_case_rows": cases},
        "reports/explainability/summary.json",
    )
    log.info("Explainability artifacts in %s; top feature: %s",
             REPORT_DIR, ranking.iloc[0]["feature"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--sample-n", type=int, default=5000)
    args = parser.parse_args()
    run(args.config, args.sample_n)
