"""Evaluation: metrics matrix, threshold tuning, statistical tests,
spatial/seasonal error decomposition, and fairness audit.

Implements blueprint Phase 4 with the validation-report fixes:
- Primary threshold = F-beta (beta=2) maximization on the PR curve (M2);
  Youden's J reported secondarily.
- Wilcoxon signed-rank on per-fold PR-AUCs and Diebold-Mariano on
  per-day Brier-score loss differentials.
- Per-zone / per-monsoon-phase decomposition for the geographic equity
  audit (Phase 5b).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------
def tune_threshold_fbeta(y_true, proba, beta: float = 2.0) -> dict:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    fbeta = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
    i = int(np.nanargmax(fbeta))
    return {"threshold": float(thr[i]), "fbeta": float(fbeta[i]),
            "precision": float(prec[i]), "recall": float(rec[i])}


def tune_threshold_youden(y_true, proba) -> dict:
    fpr, tpr, thr = roc_curve(y_true, proba)
    i = int(np.argmax(tpr - fpr))
    return {"threshold": float(thr[i]), "youden_j": float(tpr[i] - fpr[i])}


# --------------------------------------------------------------------------
# Statistical significance
# --------------------------------------------------------------------------
def wilcoxon_folds(scores_a: list[float], scores_b: list[float]) -> dict:
    """Paired Wilcoxon signed-rank on per-fold scores (small-n exact)."""
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b)
        return {"statistic": float(stat), "p_value": float(p)}
    except ValueError as exc:  # identical scores
        return {"statistic": None, "p_value": None, "note": str(exc)}


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> dict:
    """DM test on a loss differential series (HAC variance, lag ~ n^(1/3))."""
    d = loss_a - loss_b
    n = len(d)
    d_bar = d.mean()
    lag = max(1, int(np.floor(n ** (1 / 3))))
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for k in range(1, lag + 1):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        var_d += 2 * (1 - k / (lag + 1)) * cov
    dm = d_bar / np.sqrt(max(var_d, 1e-12) / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {"dm_statistic": float(dm), "p_value": float(p),
            "mean_loss_diff": float(d_bar)}


# --------------------------------------------------------------------------
# Metrics matrices
# --------------------------------------------------------------------------
def classification_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def regression_metrics(y_true, y_pred, nonzero_mask) -> dict:
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if nonzero_mask.sum() > 0:  # MAPE only on rainy days (zero-inflation)
        yt, yp = y_true[nonzero_mask], y_pred[nonzero_mask]
        out["mape_nonzero"] = float(np.mean(np.abs((yt - yp) / yt)) * 100)
    return out


def decompose_errors(df_eval: pd.DataFrame, threshold: float) -> dict:
    """Per-zone and per-monsoon-phase performance for the fairness audit."""
    out: dict = {}
    for dim in ("climate_zone", "monsoon_phase"):
        if dim not in df_eval.columns:
            continue
        out[dim] = {}
        for name, g in df_eval.groupby(dim):
            if g["y_true"].nunique() < 2:
                continue
            out[dim][str(name)] = classification_metrics(
                g["y_true"].values, g["proba"].values, threshold)
    # extreme-day miss analysis: recall on heavy-rain (> 20 mm) days
    heavy = df_eval[df_eval["precip_true"] > 20.0]
    if len(heavy) > 0:
        pred = (heavy["proba"] >= threshold).astype(int)
        out["heavy_rain_gt20mm"] = {
            "n": int(len(heavy)),
            "recall": float((pred == 1).mean()),
        }
    return out


# --------------------------------------------------------------------------
# Main: hold-out evaluation of the final tuned model
# --------------------------------------------------------------------------
def run(config_path: str = "config/config.yaml") -> dict:
    cfg = load_config(config_path)
    df = pd.read_parquet(cfg["data"]["processed_parquet"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    spec = joblib.load("models/feature_spec.joblib")
    feat_cols = spec["feat_cols"]
    target_c, target_r = cfg["target"]["classification"], cfg["target"]["regression"]

    # temporal hold-out: final 15% of unique dates (never seen in training CV)
    unique_dates = np.sort(df["time"].unique())
    cut = unique_dates[int(len(unique_dates) * 0.85)]
    val_prev = df[df["time"] < cut]
    test = df[df["time"] >= cut]
    log.info("Hold-out: %d rows from %s", len(test), str(cut)[:10])

    clf = CatBoostClassifier()
    clf.load_model(cfg["serving"]["model_path"])
    reg = CatBoostRegressor()
    reg.load_model("models/catboost_precip_amount.cbm")

    proba = clf.predict_proba(test[feat_cols])[:, 1]
    y_true = test[target_c].values

    # threshold tuned on the pre-holdout tail (last 15% of training period)
    tail_cut = unique_dates[int(len(unique_dates) * 0.70)]
    tail = val_prev[val_prev["time"] >= tail_cut]
    tail_proba = clf.predict_proba(tail[feat_cols])[:, 1]
    fbeta_sel = tune_threshold_fbeta(tail[target_c].values, tail_proba,
                                     cfg["thresholds"]["beta"])
    youden_sel = tune_threshold_youden(tail[target_c].values, tail_proba)
    thr = fbeta_sel["threshold"]

    report = {
        "holdout_start": str(cut)[:10],
        "threshold_selection": {"fbeta": fbeta_sel, "youden": youden_sel,
                                "chosen": thr, "policy": "F2 on PR curve"},
        "classification_default_0.5": classification_metrics(y_true, proba, 0.5),
        "classification_tuned": classification_metrics(y_true, proba, thr),
    }

    # hurdle regression: E[precip] = P(rain) * expm1(reg prediction)
    amount = np.expm1(reg.predict(test[feat_cols]))
    y_reg_pred = np.clip(proba * amount, 0, None)
    y_reg_true = test[target_r].values
    report["regression_hurdle"] = regression_metrics(
        y_reg_true, y_reg_pred,
        y_reg_true > cfg["target"]["rain_threshold_mm"])

    # error decomposition + fairness
    df_eval = test[["climate_zone", "monsoon_phase"]].copy()
    df_eval["y_true"], df_eval["proba"] = y_true, proba
    df_eval["precip_true"] = y_reg_true
    report["error_decomposition"] = decompose_errors(df_eval, thr)

    # statistical tests vs benchmark folds (if training report exists)
    tr_path = Path("reports/training_report.json")
    if tr_path.exists():
        bench = json.loads(tr_path.read_text())["benchmark"]
        if "catboost" in bench:
            cb = bench["catboost"]["pr_auc_folds"]
            report["significance"] = {}
            for fam, s in bench.items():
                if fam == "catboost":
                    continue
                report["significance"][f"catboost_vs_{fam}"] = wilcoxon_folds(
                    cb, s["pr_auc_folds"])
        # DM test on holdout: tuned CatBoost vs climatology baseline
        clim = np.full_like(proba, y_true.mean(), dtype=float)
        report["dm_vs_climatology"] = diebold_mariano(
            (proba - y_true) ** 2, (clim - y_true) ** 2)

    save_json(report, "reports/evaluation_report.json")
    save_json({"threshold": thr, "policy": "F2"},
              cfg["serving"]["threshold_path"])
    log.info("Tuned holdout: PR-AUC=%.4f recall=%.3f precision=%.3f",
             report["classification_tuned"]["pr_auc"],
             report["classification_tuned"]["recall"],
             report["classification_tuned"]["precision"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate final model.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)
