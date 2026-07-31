"""Training: 5 model families + LSTM, PanelTimeSeriesSplit CV, Optuna TPE.

Fixes applied vs the blueprint snippet (see VALIDATION_REPORT.md):
- C1: PanelTimeSeriesSplit splits on unique sorted DATES (expanding window,
  optional embargo), so no city's future leaks into any training fold.
- M1: class weights are the primary imbalance treatment; SMOTE is optional
  and applied strictly inside training folds.
- M3: CatBoost early stopping per fold + Optuna MedianPruner.
- M6: regression head is a hurdle model (classifier gate x rainy-day
  amount regressor on log1p scale).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils import get_logger, load_config, save_json, set_seed

log = get_logger(__name__)

try:  # MLflow tracking is optional at runtime
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    _HAS_MLFLOW = False

try:  # PyTorch LSTM benchmark is optional
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


# --------------------------------------------------------------------------
# Leakage-safe cross-validation for a multi-city panel
# --------------------------------------------------------------------------
@dataclass
class PanelTimeSeriesSplit:
    """Grouped, blocked, expanding-window split for city-date panel data.

    Splits on the ordered set of unique dates: fold k trains on the first
    (k+1)/(n_splits+1) share of dates and validates on the next block,
    with an ``embargo_days`` gap between them. Every city contributes past
    rows to train and future rows to validation — never the reverse.
    """

    n_splits: int = 5
    embargo_days: int = 1

    def split(self, dates: pd.Series):
        unique_dates = np.sort(dates.unique())
        n = len(unique_dates)
        fold_size = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            train_end = fold_size * (k + 1)
            val_start = train_end + self.embargo_days
            val_end = min(train_end + fold_size, n)
            if val_start >= val_end:
                continue
            train_dates = set(unique_dates[:train_end])
            val_dates = set(unique_dates[val_start:val_end])
            train_idx = np.flatnonzero(dates.isin(train_dates).values)
            val_idx = np.flatnonzero(dates.isin(val_dates).values)
            yield train_idx, val_idx


# --------------------------------------------------------------------------
# Model factories (5 families)
# --------------------------------------------------------------------------
def _preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def build_model(family: str, num_cols, cat_cols, seed: int, scale_pos_weight: float):
    """Return an unfitted estimator for the given family."""
    if family == "logistic":
        return Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols)),
            ("clf", LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)),
        ])
    if family == "knn":
        return Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols)),
            ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance",
                                         algorithm="kd_tree", n_jobs=-1)),
        ])
    if family == "random_forest":
        return Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols)),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=14, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        ])
    if family == "lightgbm":
        return Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols)),
            ("clf", LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=63,
                scale_pos_weight=scale_pos_weight, random_state=seed,
                n_jobs=-1, verbosity=-1)),
        ])
    if family == "xgboost":
        return Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols)),
            ("clf", XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                subsample=0.9, colsample_bytree=0.9, eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight, random_state=seed,
                n_jobs=-1, tree_method="hist")),
        ])
    if family == "mlp":  # ANN/MLP tabular variant of the deep-learning family
        from sklearn.neural_network import MLPClassifier

        return Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu",
                alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                max_iter=60, early_stopping=True, n_iter_no_change=5,
                random_state=seed)),
        ])
    if family == "catboost":
        return CatBoostClassifier(
            iterations=800, learning_rate=0.06, depth=7,
            auto_class_weights="Balanced", random_seed=seed,
            od_type="Iter", od_wait=60, verbose=0, allow_writing_files=False)
    raise ValueError(f"Unknown family: {family}")


# --------------------------------------------------------------------------
# Optional fold-internal SMOTE (M1)
# --------------------------------------------------------------------------
def maybe_resample(X_tr, y_tr, strategy: str, cat_cols: list[str], seed: int):
    if strategy != "smote":
        return X_tr, y_tr
    from imblearn.over_sampling import SMOTENC

    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    sm = SMOTENC(categorical_features=cat_idx, random_state=seed)
    X_res, y_res = sm.fit_resample(X_tr, y_tr)
    return X_res, y_res


# --------------------------------------------------------------------------
# LSTM benchmark (per-city sequences, 7-day lookback)
# --------------------------------------------------------------------------
if _HAS_TORCH:

    class RainLSTM(nn.Module):
        def __init__(self, n_features: int, hidden: int = 64):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):  # x: (B, T, F)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)


def _build_sequences(df, num_cols, target, lookback):
    """Per-city rolling windows — never crossing city boundaries."""
    xs, ys, idxs = [], [], []
    for _, g in df.groupby("city", sort=False):
        vals = g[num_cols].to_numpy(dtype=np.float32)
        tgt = g[target].to_numpy()
        for i in range(lookback - 1, len(g)):
            xs.append(vals[i - lookback + 1: i + 1])
            ys.append(tgt[i])
            idxs.append(g.index[i])
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(idxs)


def train_eval_lstm(df, num_cols, target, dates, cv, lstm_cfg, seed):
    """Blocked-CV evaluation of the LSTM benchmark. Returns fold PR-AUCs."""
    if not _HAS_TORCH:
        log.warning("PyTorch unavailable — skipping LSTM benchmark.")
        return None
    torch.manual_seed(seed)
    X_seq, y_seq, row_idx = _build_sequences(df, num_cols, target, lstm_cfg["lookback"])
    mu, sd = X_seq.mean((0, 1), keepdims=True), X_seq.std((0, 1), keepdims=True) + 1e-6
    X_seq = (X_seq - mu) / sd
    date_of_row = dates.loc[row_idx].reset_index(drop=True)
    scores = []
    for tr, va in cv.split(date_of_row):
        model = RainLSTM(X_seq.shape[-1], lstm_cfg["hidden_size"])
        pos_w = torch.tensor([(len(tr) - y_seq[tr].sum()) / max(y_seq[tr].sum(), 1)])
        opt = torch.optim.Adam(model.parameters(), lr=lstm_cfg["lr"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        Xt = torch.from_numpy(X_seq[tr])
        yt = torch.from_numpy(y_seq[tr])
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=lstm_cfg["batch_size"], shuffle=True)
        model.train()
        for _ in range(lstm_cfg["epochs"]):
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(X_seq[va]))).numpy()
        scores.append(average_precision_score(y_seq[va], p))
    return scores


# --------------------------------------------------------------------------
# Cross-validated benchmark of all families
# --------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    family: str
    pr_auc_folds: list = field(default_factory=list)
    roc_auc_folds: list = field(default_factory=list)
    fit_seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "family": self.family,
            "pr_auc_mean": float(np.mean(self.pr_auc_folds)),
            "pr_auc_std": float(np.std(self.pr_auc_folds)),
            "roc_auc_mean": float(np.mean(self.roc_auc_folds)),
            "pr_auc_folds": [float(s) for s in self.pr_auc_folds],
            "fit_seconds": round(self.fit_seconds, 1),
        }


def benchmark_families(df, feat_cols, cat_cols, cfg) -> dict[str, BenchmarkResult]:
    target = cfg["target"]["classification"]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    X, y, dates = df[feat_cols], df[target], df["time"]
    pos_rate = float(y.mean())
    spw = (1 - pos_rate) / max(pos_rate, 1e-6)
    cv = PanelTimeSeriesSplit(cfg["cv"]["n_splits"], cfg["cv"]["embargo_days"])
    knn_cap = cfg["models"]["knn_max_train_rows"]
    results: dict[str, BenchmarkResult] = {}

    for family in cfg["models"]["families"]:
        if family == "lstm":
            t0 = time.time()
            folds = train_eval_lstm(df, num_cols, target, dates, cv,
                                    cfg["models"]["lstm"], cfg["seed"])
            if folds is not None:
                res = BenchmarkResult("lstm", folds, [float("nan")] * len(folds),
                                      time.time() - t0)
                results["lstm"] = res
                log.info("lstm: PR-AUC %.4f +/- %.4f", np.mean(folds), np.std(folds))
            continue
        res = BenchmarkResult(family)
        t0 = time.time()
        for tr, va in cv.split(dates):
            if family == "knn" and len(tr) > knn_cap:  # tractability cap
                rng = np.random.default_rng(cfg["seed"])
                tr = rng.choice(tr, size=knn_cap, replace=False)
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            X_tr, y_tr = maybe_resample(
                X_tr, y_tr, cfg["imbalance"]["strategy"], cat_cols, cfg["seed"])
            model = build_model(family, num_cols, cat_cols, cfg["seed"], spw)
            if family == "catboost":
                model.fit(X_tr, y_tr, cat_features=cat_cols)
            else:
                model.fit(X_tr, y_tr)
            proba = model.predict_proba(X.iloc[va])[:, 1]
            res.pr_auc_folds.append(average_precision_score(y.iloc[va], proba))
            res.roc_auc_folds.append(roc_auc_score(y.iloc[va], proba))
        res.fit_seconds = time.time() - t0
        results[family] = res
        log.info("%s: PR-AUC %.4f +/- %.4f", family,
                 np.mean(res.pr_auc_folds), np.std(res.pr_auc_folds))
    return results


# --------------------------------------------------------------------------
# Optuna TPE tuning of the CatBoost frontrunner
# --------------------------------------------------------------------------
def tune_catboost(df, feat_cols, cat_cols, cfg) -> optuna.Study:
    target = cfg["target"]["classification"]
    X, y, dates = df[feat_cols], df[target], df["time"]
    cv = PanelTimeSeriesSplit(cfg["cv"]["n_splits"], cfg["cv"]["embargo_days"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            "auto_class_weights": "Balanced",
            "random_seed": cfg["seed"],
            "od_type": "Iter", "od_wait": 60,
            "verbose": 0, "allow_writing_files": False,
        }
        scores = []
        for k, (tr, va) in enumerate(cv.split(dates)):
            model = CatBoostClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_cols,
                      eval_set=(X.iloc[va], y.iloc[va]))
            score = average_precision_score(
                y.iloc[va], model.predict_proba(X.iloc[va])[:, 1])
            scores.append(score)
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"],
                   timeout=cfg["optuna"]["timeout_seconds"])
    log.info("Optuna best PR-AUC=%.4f params=%s", study.best_value, study.best_params)
    return study


# --------------------------------------------------------------------------
# Final fit: tuned classifier + hurdle regression head
# --------------------------------------------------------------------------
def fit_final(df, feat_cols, cat_cols, cfg, best_params: dict) -> dict:
    target_c = cfg["target"]["classification"]
    target_r = cfg["target"]["regression"]
    X, y_c = df[feat_cols], df[target_c]

    clf = CatBoostClassifier(
        **best_params, auto_class_weights="Balanced", random_seed=cfg["seed"],
        verbose=0, allow_writing_files=False)
    clf.fit(X, y_c, cat_features=cat_cols)

    rainy = df[df[target_r] > cfg["target"]["rain_threshold_mm"]]
    reg = CatBoostRegressor(
        depth=best_params.get("depth", 7),
        learning_rate=best_params.get("learning_rate", 0.06),
        iterations=best_params.get("iterations", 800),
        loss_function="RMSE", random_seed=cfg["seed"],
        verbose=0, allow_writing_files=False)
    reg.fit(rainy[feat_cols], np.log1p(rainy[target_r]), cat_features=cat_cols)

    Path("models").mkdir(exist_ok=True)
    clf.save_model(cfg["serving"]["model_path"])
    reg.save_model("models/catboost_precip_amount.cbm")
    joblib.dump({"feat_cols": feat_cols, "cat_cols": cat_cols},
                "models/feature_spec.joblib")
    return {"classifier": cfg["serving"]["model_path"],
            "regressor": "models/catboost_precip_amount.cbm"}


def run(config_path: str = "config/config.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    df = pd.read_parquet(cfg["data"]["processed_parquet"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    cat_cols = [c for c in ("city", "monsoon_phase", "climate_zone") if c in df.columns]
    feat_cols = [c for c in df.columns
                 if c not in ("time", cfg["target"]["classification"],
                              cfg["target"]["regression"])]

    if _HAS_MLFLOW:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])

    def _log_all(results, study):
        summaries = {f: r.summary() for f, r in results.items()}
        save_json({"benchmark": summaries,
                   "optuna_best_value": study.best_value,
                   "optuna_best_params": study.best_params},
                  "reports/training_report.json")
        if _HAS_MLFLOW:
            with mlflow.start_run(run_name="benchmark"):
                for fam, s in summaries.items():
                    mlflow.log_metric(f"{fam}_pr_auc_mean", s["pr_auc_mean"])
                mlflow.log_metric("catboost_tuned_pr_auc", study.best_value)
                mlflow.log_params(
                    {f"best_{k}": v for k, v in study.best_params.items()})
                mlflow.log_artifact("reports/training_report.json")

    results = benchmark_families(df, feat_cols, cat_cols, cfg)
    study = tune_catboost(df, feat_cols, cat_cols, cfg)
    _log_all(results, study)
    paths = fit_final(df, feat_cols, cat_cols, cfg, study.best_params)
    log.info("Final models saved: %s", paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, benchmark, tune.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)
