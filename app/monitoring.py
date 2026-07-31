"""Drift monitoring: Evidently report + statistical fallback.

Compares production feature/prediction distributions against the training
reference using KS (numeric) and Chi-square (categorical) at alpha=0.05,
per the blueprint. Uses Evidently's Report API when available; otherwise a
scipy implementation of the same tests, so drift detection is guaranteed
to run in any environment. Exits non-zero on alert so a scheduler (Cloud
Scheduler / cron / Actions) can chain the retraining job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)


def _scipy_drift(reference: pd.DataFrame, current: pd.DataFrame,
                 alpha: float) -> dict:
    results = {}
    for col in reference.columns:
        if col not in current.columns:
            continue
        ref, cur = reference[col].dropna(), current[col].dropna()
        if ref.empty or cur.empty:
            continue
        if pd.api.types.is_numeric_dtype(ref):
            stat, p = stats.ks_2samp(ref, cur)
            test = "ks"
        else:
            cats = sorted(set(ref.unique()) | set(cur.unique()))
            ref_c = ref.value_counts().reindex(cats, fill_value=0)
            cur_c = cur.value_counts().reindex(cats, fill_value=0)
            expected = ref_c / ref_c.sum() * cur_c.sum()
            mask = expected > 0
            stat, p = stats.chisquare(cur_c[mask], expected[mask])
            test = "chi2"
        results[col] = {"test": test, "statistic": float(stat),
                        "p_value": float(p), "drift": bool(p < alpha)}
    return results


def _evidently_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict | None:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        out = Path("reports/drift/evidently_report.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(out))
        return {"evidently_html": str(out)}
    except Exception as exc:  # version drift in Evidently API itself
        log.warning("Evidently unavailable/failed (%s); scipy fallback only.", exc)
        return None


def run(config_path: str = "config/config.yaml",
        current_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    alpha = cfg["monitoring"]["drift_alpha"]
    share_alert = cfg["monitoring"]["drift_share_alert"]

    df = pd.read_parquet(cfg["data"]["processed_parquet"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    drift_cols = [c for c in df.columns
                  if c not in ("time", cfg["target"]["classification"],
                               cfg["target"]["regression"])]

    reference = df.iloc[: int(len(df) * 0.85)][drift_cols]
    if current_path:  # production window (e.g. parsed prediction log)
        current = pd.read_parquet(current_path)[drift_cols]
    else:  # default demo: latest 15% simulates the production window
        current = df.iloc[int(len(df) * 0.85):][drift_cols]

    per_feature = _scipy_drift(reference, current, alpha)
    n_drifted = sum(v["drift"] for v in per_feature.values())
    share = n_drifted / max(len(per_feature), 1)
    alert = share >= share_alert

    report = {
        "alpha": alpha, "n_features": len(per_feature),
        "n_drifted": n_drifted, "drift_share": round(share, 3),
        "alert": alert, "per_feature": per_feature,
    }
    ev = _evidently_drift(reference, current)
    if ev:
        report.update(ev)
    save_json(report, "reports/drift/drift_report.json")
    log.info("Drift: %d/%d features (share=%.2f) -> alert=%s",
             n_drifted, len(per_feature), share, alert)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift monitoring.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--current", default=None,
                        help="Parquet of production window (optional)")
    args = parser.parse_args()
    result = run(args.config, args.current)
    sys.exit(1 if result["alert"] else 0)
