"""Shared utilities: config loading, logging, seeding, zone mapping."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def assign_climate_zone(df: pd.DataFrame) -> pd.Series:
    """Assign agro-climatic zone from elevation and geography.

    Data-driven rule (robust to the exact 30-city roster):
    - hill country : elevation > 500 m (Kandy, Nuwara Eliya, Badulla, ...)
    - wet zone     : southwest lowlands (lat <= 7.6 and lon <= 80.6)
    - dry zone     : remainder (northern/eastern lowlands)
    """
    zone = pd.Series("dry_zone", index=df.index, dtype="object")
    wet = (df["latitude"] <= 7.6) & (df["longitude"] <= 80.6)
    zone[wet] = "wet_zone"
    zone[df["elevation"] > 500] = "hill_country"
    return zone


def assign_monsoon_phase(dates: pd.Series) -> pd.Series:
    """Map calendar month to Sri Lankan monsoon phase.

    SW (Yala) May-Sep, NE (Maha) Dec-Feb, inter-monsoon 1 Mar-Apr,
    inter-monsoon 2 Oct-Nov.
    """
    month = pd.to_datetime(dates).dt.month
    phase = pd.Series("inter1", index=dates.index, dtype="object")
    phase[month.isin([5, 6, 7, 8, 9])] = "SW"
    phase[month.isin([12, 1, 2])] = "NE"
    phase[month.isin([3, 4])] = "inter1"
    phase[month.isin([10, 11])] = "inter2"
    return phase
