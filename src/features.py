"""Monsoon-aware feature engineering and three-stage feature selection.

Implements the blueprint's six novel features:
1. Rolling standardized climatic anomaly  (precip - mu_w) / sigma_w
2. NWS Steadman heat-index proxy with RH approximated from the
   apparent-vs-actual temperature spread
3. Cyclical month encoding + categorical monsoon phase
4. Diurnal temperature range (DTR)
5. Wind vector components u = ws*cos(theta), v = ws*sin(theta)
6. Aridity / water-balance ratio  precip / (ET0 + eps)

plus per-city lag features and the leakage-safe next-day targets.

Stability fixes vs the blueprint snippet (VALIDATION_REPORT M5):
sigma floored at 1e-6 AND anomalies clipped to +/- anomaly_clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    assign_climate_zone,
    assign_monsoon_phase,
    get_logger,
    load_config,
    save_json,
)

log = get_logger(__name__)

CATEGORICAL_FEATURES = ["city", "monsoon_phase", "climate_zone"]
DROP_ALWAYS = ["time", "country", "sunrise", "sunset", "weather_code"]


def _heat_index_f(temp_c: pd.Series, apparent_c: pd.Series) -> tuple[pd.Series, pd.Series]:
    """NWS/Steadman heat-index regression with RH proxied from the
    apparent-actual temperature spread (clipped to [20, 100] percent).

    Returns (heat_index_f, rh_proxy) so the proxy itself is inspectable
    downstream by SHAP (VALIDATION_REPORT M4).
    """
    t_f = temp_c * 9.0 / 5.0 + 32.0
    rh = (100.0 - 5.0 * (temp_c - apparent_c).abs()).clip(20.0, 100.0)
    hi = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh
        - 0.22475541 * t_f * rh - 6.83783e-3 * t_f**2
        - 5.481717e-2 * rh**2 + 1.22874e-3 * t_f**2 * rh
        + 8.5282e-4 * t_f * rh**2 - 1.99e-6 * t_f**2 * rh**2
    )
    return hi, rh


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build all engineered features, lags, and next-day targets."""
    clip = float(cfg["features"]["anomaly_clip"])
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    grp = df.groupby("city", group_keys=False)

    # 1. rolling standardized climatic anomaly (7/15/30-day)
    for w in cfg["features"]["anomaly_windows"]:
        mu = grp["precipitation_sum"].transform(
            lambda s: s.rolling(w, min_periods=3).mean()
        )
        sd = grp["precipitation_sum"].transform(
            lambda s: s.rolling(w, min_periods=3).std()
        )
        df[f"precip_anom_{w}d"] = (
            (df["precipitation_sum"] - mu) / (sd + 1e-6)
        ).clip(-clip, clip)

    # 2. heat index + RH proxy
    df["heat_index_f"], df["rh_proxy"] = _heat_index_f(
        df["temperature_2m_mean"], df["apparent_temperature_mean"]
    )

    # 3. cyclical month + monsoon phase + climate zone
    month = df["time"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["monsoon_phase"] = assign_monsoon_phase(df["time"])
    df["climate_zone"] = assign_climate_zone(df)

    # 4. diurnal temperature range
    df["dtr"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # 5. wind vector components (meteorological direction -> Cartesian)
    theta = np.deg2rad(df["winddirection_10m_dominant"])
    df["wind_u"] = df["windspeed_10m_max"] * np.cos(theta)
    df["wind_v"] = df["windspeed_10m_max"] * np.sin(theta)

    # 6. water balance / aridity ratio
    df["precip_to_et"] = df["precipitation_sum"] / (
        df["et0_fao_evapotranspiration"] + 1e-6
    )

    # per-city lags of precipitation and key drivers
    for lag in cfg["features"]["lag_days"]:
        df[f"precip_lag{lag}"] = grp["precipitation_sum"].shift(lag)
    df["rain_today"] = (
        df["precipitation_sum"] > cfg["target"]["rain_threshold_mm"]
    ).astype(np.int8)
    df["rain_yesterday"] = grp["rain_today"].shift(1)

    # leakage-safe next-day targets (features at t, targets at t+1)
    df["precip_tomorrow"] = grp["precipitation_sum"].shift(-1)
    df["rain_tomorrow"] = (
        df["precip_tomorrow"] > cfg["target"]["rain_threshold_mm"]
    ).astype("Int8")

    # drop rows lacking full lag history or a next-day target
    lag_cols = [f"precip_lag{lag}" for lag in cfg["features"]["lag_days"]]
    df = df.dropna(subset=lag_cols + ["precip_tomorrow", "rain_yesterday"]).copy()
    df["rain_tomorrow"] = df["rain_tomorrow"].astype(np.int8)
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """All model-input columns (targets and raw leak-prone cols excluded)."""
    exclude = set(DROP_ALWAYS) | {
        cfg["target"]["classification"], cfg["target"]["regression"],
    }
    return [c for c in df.columns if c not in exclude]


def correlation_filter(
    df: pd.DataFrame, cols: list[str], threshold: float
) -> tuple[list[str], list[str]]:
    """Stage-1 filter: drop one of each numeric pair with |r| > threshold."""
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    # drop zero-variance columns first (e.g. snowfall_sum in Sri Lanka)
    constant = [c for c in numeric if df[c].nunique(dropna=True) <= 1]
    numeric = [c for c in numeric if c not in constant]
    cols = [c for c in cols if c not in constant]
    corr = df[numeric].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    kept = [c for c in cols if c not in dropped]
    return kept, dropped + constant


def run(config_path: str = "config/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    clean_path = Path(cfg["data"]["processed_dir"]) / "clean.parquet"
    df = pd.read_parquet(clean_path)
    df["time"] = pd.to_datetime(df["time"])
    df = engineer_features(df, cfg)

    cols = feature_columns(df, cfg)
    kept, dropped = correlation_filter(
        df, cols, cfg["features"]["corr_filter_threshold"]
    )
    save_json(
        {
            "n_rows": int(len(df)),
            "n_features_before_filter": len(cols),
            "n_features_after_filter": len(kept),
            "dropped_correlated": dropped,
            "kept": kept,
            "class_balance_rain_tomorrow": float(df["rain_tomorrow"].mean()),
        },
        "reports/feature_report.json",
    )
    out = Path(cfg["data"]["processed_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = kept + ["time", cfg["target"]["classification"], cfg["target"]["regression"]]
    df[[c for c in dict.fromkeys(keep_cols)]].to_parquet(out, index=False)
    log.info(
        "Features written: %s (%d rows, %d model features; dropped %s)",
        out, len(df), len(kept), dropped,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)
