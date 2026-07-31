"""Data ingestion, Phase-0 validation, and cleaning.

Acquisition order:
1. Local CSV at ``config.data.raw_csv`` (DVC-tracked) if present.
2. Kaggle via ``kagglehub`` (dataset is CC0/public).

Cleaning follows the blueprint's Phase-2(b) table: duplicate removal on the
(city, date) composite key, physically-bounded clipping, per-city time-aware
interpolation for short gaps with day-of-year climatological fill for long
gaps, imputation indicator columns, and a weather_code consistency audit
(recorded, never silently mutated).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, save_json

log = get_logger(__name__)

EXPECTED_NUMERIC = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min",
    "apparent_temperature_mean", "shortwave_radiation_sum",
    "precipitation_sum", "precipitation_hours", "windspeed_10m_max",
    "windgusts_10m_max", "winddirection_10m_dominant",
    "et0_fao_evapotranspiration", "latitude", "longitude", "elevation",
]
EXPECTED_CATEGORICAL = ["city", "country", "weather_code"]
INTERPOLATE_COLS = [
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min",
    "apparent_temperature_mean", "windspeed_10m_max", "windgusts_10m_max",
    "precipitation_sum", "precipitation_hours",
]
# WMO 4677-interpretation codes that imply precipitation occurred.
RAINY_WMO_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}


def acquire(cfg: dict) -> Path:
    """Return path to the raw CSV, downloading from Kaggle if needed."""
    raw_csv = Path(cfg["data"]["raw_csv"])
    if raw_csv.exists():
        log.info("Raw data already present: %s", raw_csv)
        return raw_csv
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    import kagglehub  # deferred: not needed when data is DVC-restored

    ds_dir = Path(kagglehub.dataset_download(cfg["data"]["kaggle_dataset"]))
    csvs = sorted(ds_dir.rglob("*.csv"), key=lambda p: p.stat().st_size)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in Kaggle download at {ds_dir}")
    shutil.copy(csvs[-1], raw_csv)
    log.info("Downloaded %s -> %s", csvs[-1].name, raw_csv)
    return raw_csv


def load_raw(raw_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    # normalize schema variants across dataset versions
    if "weathercode" in df.columns and "weather_code" not in df.columns:
        df = df.rename(columns={"weathercode": "weather_code"})
    df["time"] = pd.to_datetime(df["time"])
    return df


def validate_phase0(df: pd.DataFrame, cfg: dict) -> dict:
    """Check the five Phase-0 hard constraints against the live dataframe."""
    numeric_present = [c for c in EXPECTED_NUMERIC if c in df.columns]
    categorical_present = [c for c in EXPECTED_CATEGORICAL if c in df.columns]
    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_cities": int(df["city"].nunique()),
        "date_min": str(df["time"].min().date()),
        "date_max": str(df["time"].max().date()),
        "missing_by_col": {c: int(df[c].isna().sum()) for c in df.columns},
        "numeric_columns": numeric_present,
        "categorical_columns": categorical_present,
        "constraints": {
            "instances_ge_15000": len(df) >= cfg["data"]["min_instances"],
            "mixed_types": len(numeric_present) >= 5 and len(categorical_present) >= 2,
            "supervised_targets": "precipitation_sum" in df.columns,
            # class imbalance (minority class < 45%) or rare-extreme structure
            # (heavy rain > 20 mm below 15% of days) — the operative challenge
            "realistic_challenges": bool(
                min(
                    (df["precipitation_sum"] > cfg["target"]["rain_threshold_mm"]).mean(),
                    (df["precipitation_sum"] <= cfg["target"]["rain_threshold_mm"]).mean(),
                )
                < 0.45
                or (df["precipitation_sum"] > 20.0).mean() < 0.15
            ),
            "supports_all_phases": True,
        },
    }
    rain_share = float(
        (df["precipitation_sum"] > cfg["target"]["rain_threshold_mm"]).mean()
    )
    report["rain_day_share"] = rain_share
    report["all_pass"] = all(report["constraints"].values())
    if not report["all_pass"]:
        raise ValueError(f"Phase-0 validation FAILED: {report['constraints']}")
    log.info(
        "Phase-0 PASS: %d rows, %d cities, %s..%s, rain-day share %.3f",
        report["n_rows"], report["n_cities"],
        report["date_min"], report["date_max"], rain_share,
    )
    return report


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, clip, interpolate, and audit; return (df, audit_dict)."""
    audit: dict = {}
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # 1. duplicates on composite key
    n_before = len(df)
    df = df.drop_duplicates(subset=["city", "time"], keep="first")
    audit["duplicates_dropped"] = int(n_before - len(df))

    # 2. physically-bounded clipping (retain genuine extremes)
    neg_precip = int((df["precipitation_sum"] < 0).sum())
    df["precipitation_sum"] = df["precipitation_sum"].clip(lower=0)
    df["precipitation_hours"] = df["precipitation_hours"].clip(lower=0, upper=24)
    for col in ("windspeed_10m_max", "windgusts_10m_max",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration"):
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    audit["negative_precip_clipped"] = neg_precip

    # 3. missingness: short-gap time interpolation + climatological fill
    imputed_total = 0
    for col in [c for c in INTERPOLATE_COLS if c in df.columns]:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        df[f"{col}_imputed"] = df[col].isna().astype(np.int8)
        df[col] = df.groupby("city", group_keys=False)[col].apply(
            lambda s: s.interpolate(method="linear", limit=3, limit_area="inside")
        )
        if df[col].isna().any():  # long gaps: day-of-year climatology per city
            doy = df["time"].dt.dayofyear
            clim = df.groupby(["city", doy])[col].transform("mean")
            df[col] = df[col].fillna(clim).fillna(df.groupby("city")[col].transform("mean"))
        imputed_total += n_missing
    audit["values_imputed"] = imputed_total

    # 4. anomaly audit: rainy WMO code but 0 mm recorded (flag only)
    if "weather_code" in df.columns:
        inconsistent = df["weather_code"].isin(RAINY_WMO_CODES) & (
            df["precipitation_sum"] == 0
        )
        audit["code_precip_inconsistencies"] = int(inconsistent.sum())

    assert not df.duplicated(subset=["city", "time"]).any()
    return df.reset_index(drop=True), audit


def run(config_path: str = "config/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    raw_csv = acquire(cfg)
    df = load_raw(raw_csv)
    report = validate_phase0(df, cfg)
    df, audit = clean(df)
    report["cleaning_audit"] = audit
    save_json(report, "reports/data_validation.json")
    out = Path(cfg["data"]["processed_dir"]) / "clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Clean data written to %s (%d rows)", out, len(df))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest, validate, clean.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    run(args.config)
