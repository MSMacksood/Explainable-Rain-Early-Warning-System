"""Chunked, resumable execution driver for constrained environments.

Runs one unit of work per invocation and checkpoints to disk, so the full
benchmark/tuning pipeline can be executed as a sequence of short steps
(CI job matrices, spot instances, or sandboxed shells with hard timeouts).

Usage:
  python scripts/run_step.py bench --family catboost --fold 0
  python scripts/run_step.py optuna --n-trials 2
  python scripts/run_step.py finalize            # aggregate + fit final clf
  python scripts/run_step.py finalize-reg        # fit hurdle regressor
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.train import PanelTimeSeriesSplit, build_model, maybe_resample
from src.utils import get_logger, load_config, save_json, set_seed

log = get_logger("run_step")
PARTIAL = Path(os.environ.get("PARTIAL_DIR", "reports/partial"))


def _load(cfg):
    df = pd.read_parquet(cfg["data"]["processed_parquet"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    cat_cols = [c for c in ("city", "monsoon_phase", "climate_zone")
                if c in df.columns]
    feat_cols = [c for c in df.columns
                 if c not in ("time", cfg["target"]["classification"],
                              cfg["target"]["regression"])]
    return df, feat_cols, cat_cols


def bench(cfg, family: str, fold: int) -> None:
    set_seed(cfg["seed"])
    df, feat_cols, cat_cols = _load(cfg)
    target = cfg["target"]["classification"]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    X, y, dates = df[feat_cols], df[target], df["time"]
    pos = float(y.mean())
    spw = (1 - pos) / max(pos, 1e-6)
    cv = PanelTimeSeriesSplit(cfg["cv"]["n_splits"], cfg["cv"]["embargo_days"])
    folds = list(cv.split(dates))
    tr, va = folds[fold]
    if family == "knn":
        cap = cfg["models"]["knn_max_train_rows"]
        if len(tr) > cap:
            tr = np.random.default_rng(cfg["seed"]).choice(tr, cap, replace=False)
    t0 = time.time()
    X_tr, y_tr = maybe_resample(X.iloc[tr], y.iloc[tr],
                                cfg["imbalance"]["strategy"], cat_cols, cfg["seed"])
    model = build_model(family, num_cols, cat_cols, cfg["seed"], spw)
    if family == "catboost":
        # eval_set enables od_type=Iter early stopping (same protocol as
        # the Optuna objective), keeping per-fold cost bounded
        model.fit(X_tr, y_tr, cat_features=cat_cols,
                  eval_set=(X.iloc[va], y.iloc[va]))
    else:
        model.fit(X_tr, y_tr)
    proba = model.predict_proba(X.iloc[va])[:, 1]
    rec = {"fold": fold,
           "pr_auc": float(average_precision_score(y.iloc[va], proba)),
           "roc_auc": float(roc_auc_score(y.iloc[va], proba)),
           "seconds": round(time.time() - t0, 1)}
    PARTIAL.mkdir(parents=True, exist_ok=True)
    out = PARTIAL / f"bench_{family}.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    log.info("%s fold %d: PR-AUC=%.4f (%.1fs)", family, fold,
             rec["pr_auc"], rec["seconds"])


def run_optuna(cfg, n_trials: int) -> None:
    import optuna
    from catboost import CatBoostClassifier

    set_seed(cfg["seed"])
    df, feat_cols, cat_cols = _load(cfg)
    target = cfg["target"]["classification"]
    X, y, dates = df[feat_cols], df[target], df["time"]
    cv = PanelTimeSeriesSplit(cfg["cv"]["n_splits"], cfg["cv"]["embargo_days"])
    folds = list(cv.split(dates))

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "iterations": trial.suggest_int("iterations", 200, 600),
            "auto_class_weights": "Balanced", "random_seed": cfg["seed"],
            "od_type": "Iter", "od_wait": 40, "verbose": 0,
            "allow_writing_files": False,
        }
        scores = []
        for k, (tr, va) in enumerate(folds):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_cols,
                  eval_set=(X.iloc[va], y.iloc[va]))
            scores.append(average_precision_score(
                y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    storage = os.environ.get("OPTUNA_STORAGE", "sqlite:///reports/optuna.db")
    study = optuna.create_study(
        study_name="catboost_tpe", storage=storage,
        load_if_exists=True, direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    # Seed derived from trial count: reproducible overall, yet each resumed
    # process continues the TPE sequence instead of repeating proposals.
    study.sampler = optuna.samplers.TPESampler(
        seed=cfg["seed"] + len(study.trials))
    study.optimize(objective, n_trials=n_trials)
    log.info("Trials done: %d | best=%.4f", len(study.trials), study.best_value)


def finalize(cfg) -> None:
    import optuna
    from catboost import CatBoostClassifier

    df, feat_cols, cat_cols = _load(cfg)
    bench_summary = {}
    for f in sorted(PARTIAL.glob("bench_*.jsonl")):
        fam = f.stem.replace("bench_", "")
        raw = [json.loads(line) for line in f.read_text().splitlines()]
        recs = list({r["fold"]: r for r in raw}.values())  # keep last per fold
        recs.sort(key=lambda r: r["fold"])
        pr = [r["pr_auc"] for r in recs]
        roc = [r["roc_auc"] for r in recs]
        bench_summary[fam] = {
            "family": fam, "pr_auc_mean": float(np.mean(pr)),
            "pr_auc_std": float(np.std(pr)),
            "roc_auc_mean": float(np.nanmean(roc)),
            "pr_auc_folds": pr,
            "fit_seconds": float(np.sum([r["seconds"] for r in recs])),
        }
    storage = os.environ.get("OPTUNA_STORAGE", "sqlite:///reports/optuna.db")
    study = optuna.load_study(study_name="catboost_tpe", storage=storage)
    save_json({"benchmark": bench_summary,
               "optuna_best_value": study.best_value,
               "optuna_best_params": study.best_params,
               "optuna_n_trials": len(study.trials)},
              "reports/training_report.json")

    clf = CatBoostClassifier(
        **study.best_params, auto_class_weights="Balanced",
        random_seed=cfg["seed"], verbose=0, allow_writing_files=False)
    X, y = df[feat_cols], df[cfg["target"]["classification"]]
    clf.fit(X, y, cat_features=cat_cols)
    Path("models").mkdir(exist_ok=True)
    clf.save_model(cfg["serving"]["model_path"])
    joblib.dump({"feat_cols": feat_cols, "cat_cols": cat_cols},
                "models/feature_spec.joblib")
    log.info("Final classifier saved; benchmark of %d families aggregated.",
             len(bench_summary))


def finalize_reg(cfg) -> None:
    import optuna
    from catboost import CatBoostRegressor

    df, feat_cols, cat_cols = _load(cfg)
    storage = os.environ.get("OPTUNA_STORAGE", "sqlite:///reports/optuna.db")
    study = optuna.load_study(study_name="catboost_tpe", storage=storage)
    p = study.best_params
    target_r = cfg["target"]["regression"]
    rainy = df[df[target_r] > cfg["target"]["rain_threshold_mm"]]
    reg = CatBoostRegressor(
        depth=p.get("depth", 7), learning_rate=p.get("learning_rate", 0.06),
        iterations=p.get("iterations", 500), loss_function="RMSE",
        random_seed=cfg["seed"], verbose=0, allow_writing_files=False)
    reg.fit(rainy[feat_cols], np.log1p(rainy[target_r]), cat_features=cat_cols)
    reg.save_model("models/catboost_precip_amount.cbm")
    log.info("Hurdle regressor saved (%d rainy rows).", len(rainy))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "optuna", "finalize", "finalize-reg"])
    ap.add_argument("--family", default="catboost")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=2)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode == "bench":
        bench(cfg, args.family, args.fold)
    elif args.mode == "optuna":
        run_optuna(cfg, args.n_trials)
    elif args.mode == "finalize":
        finalize(cfg)
    else:
        finalize_reg(cfg)
