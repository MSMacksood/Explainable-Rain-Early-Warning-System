"""Generate a physically-realistic SURROGATE of the Sri Lanka Weather Dataset.

PURPOSE: demonstration only, for environments without network access to
Kaggle/Open-Meteo. Schema, scale (30 cities x 2010-2023 daily), monsoon
seasonality, spatial zone structure, and zero-inflated precipitation match
the real dataset's documented properties. Replace with the real CSV at
``data/raw/SriLanka_Weather_Dataset.csv`` for scientific results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (city, lat, lon, elev_m, zone) — 30 Sri Lankan cities across the
# wet zone, dry zone, and hill country.
CITIES = [
    ("Colombo", 6.93, 79.85, 7, "wet"), ("Dehiwala-Mount Lavinia", 6.83, 79.87, 8, "wet"),
    ("Moratuwa", 6.77, 79.88, 5, "wet"), ("Sri Jayewardenepura Kotte", 6.90, 79.91, 8, "wet"),
    ("Negombo", 7.21, 79.84, 6, "wet"), ("Kandy", 7.30, 80.64, 500, "hill"),
    ("Kalmunai", 7.42, 81.82, 3, "dry"), ("Vavuniya", 8.75, 80.50, 105, "dry"),
    ("Galle", 6.05, 80.22, 13, "wet"), ("Trincomalee", 8.57, 81.23, 8, "dry"),
    ("Batticaloa", 7.71, 81.69, 3, "dry"), ("Jaffna", 9.67, 80.01, 5, "dry"),
    ("Katunayake", 7.17, 79.89, 8, "wet"), ("Dambulla", 7.86, 80.65, 155, "dry"),
    ("Kolonnawa", 6.93, 79.88, 6, "wet"), ("Anuradhapura", 8.31, 80.40, 81, "dry"),
    ("Ratnapura", 6.68, 80.40, 34, "wet"), ("Badulla", 6.98, 81.05, 680, "hill"),
    ("Matara", 5.95, 80.54, 5, "wet"), ("Puttalam", 8.03, 79.83, 2, "dry"),
    ("Chavakachcheri", 9.66, 80.15, 5, "dry"), ("Kurunegala", 7.49, 80.36, 116, "dry"),
    ("Mannar", 8.98, 79.90, 4, "dry"), ("Nuwara Eliya", 6.97, 80.77, 1868, "hill"),
    ("Hambantota", 6.12, 81.12, 8, "dry"), ("Kalutara", 6.58, 79.96, 5, "wet"),
    ("Matale", 7.47, 80.62, 364, "hill"), ("Gampaha", 7.09, 79.99, 12, "wet"),
    ("Ampara", 7.30, 81.68, 40, "dry"), ("Polonnaruwa", 7.94, 81.00, 47, "dry"),
]

# Monthly mean rainfall (mm/day) by zone — approximates SL climatology
# (wet zone bimodal SW-monsoon peaks; dry zone NE-monsoon peak; hill mixed).
ZONE_MONTHLY_PRECIP = {
    "wet":  [3.0, 2.5, 4.5, 8.0, 11.0, 7.5, 4.5, 4.5, 7.0, 11.5, 10.5, 5.5],
    "dry":  [5.5, 2.5, 2.0, 3.5, 2.5, 0.8, 1.2, 1.5, 2.0, 5.5, 9.5, 10.0],
    "hill": [4.0, 2.5, 3.0, 6.0, 6.5, 4.5, 3.5, 3.5, 4.5, 8.5, 8.0, 6.5],
}
WMO_DRY, WMO_LIGHT, WMO_RAIN, WMO_HEAVY = 1, 61, 63, 95


def simulate_city(rng, name, lat, lon, elev, zone, dates):
    n = len(dates)
    month = dates.month.values
    doy = dates.dayofyear.values

    # temperature: coastal ~27-30C mean, lapse rate -6.5C/km
    base_t = 27.5 - 6.5 * elev / 1000.0 + 1.2 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    t_mean = base_t + rng.normal(0, 0.9, n)

    # precipitation: gamma-mixture with Markov wet/dry persistence
    clim = np.array(ZONE_MONTHLY_PRECIP[zone])[month - 1]
    p_wet = np.clip(0.18 + 0.055 * clim, 0.1, 0.85)
    wet = np.zeros(n, dtype=bool)
    wet[0] = rng.random() < p_wet[0]
    for i in range(1, n):  # persistence: wet spells cluster
        p = p_wet[i] + (0.22 if wet[i - 1] else -0.06)
        wet[i] = rng.random() < np.clip(p, 0.03, 0.95)
    amount = rng.gamma(shape=0.85, scale=clim / 0.85 * 2.2, size=n)
    extreme = rng.random(n) < 0.012  # convective/cyclonic extremes
    amount[extreme] *= rng.uniform(3, 9, extreme.sum())
    precip = np.where(wet, np.maximum(amount, 0.1), 0.0).round(2)

    cloud = np.clip(precip / 25.0, 0, 1)
    dtr_base = {"wet": 7.5, "dry": 9.5, "hill": 9.0}[zone]
    dtr = np.clip(dtr_base - 4.0 * cloud + rng.normal(0, 0.8, n), 2.0, 14.0)
    rad = np.clip(22.0 - 9.0 * cloud + 3.0 * np.sin(2 * np.pi * (doy - 80) / 365.25)
                  + rng.normal(0, 1.5, n), 6.0, 30.0)
    et0 = np.clip(0.0023 * rad * (t_mean + 17.8) * np.sqrt(np.maximum(dtr, 0.1)) * 6.0
                  + rng.normal(0, 0.2, n), 0.5, 9.0)

    # wind: SW-monsoon flow May-Sep (~225 deg), NE Dec-Feb (~45 deg)
    wdir = np.where(np.isin(month, [5, 6, 7, 8, 9]), 225.0,
                    np.where(np.isin(month, [12, 1, 2]), 45.0, 135.0))
    wdir = (wdir + rng.normal(0, 30, n)) % 360
    wspd = np.clip(rng.gamma(4.0, 3.2, n) + 4.0 * (precip > 20), 2, 70)

    rh_eff = np.clip(70 + 18 * cloud + rng.normal(0, 4, n), 45, 100)
    app_offset = np.clip((rh_eff - 60) * 0.08 - wspd * 0.03, -3, 4)

    code = np.select(
        [precip > 20, precip > 5, precip > 0.1], [WMO_HEAVY, WMO_RAIN, WMO_LIGHT], WMO_DRY
    )
    return pd.DataFrame({
        "time": dates, "city": name, "country": "Sri Lanka", "weather_code": code,
        "temperature_2m_max": (t_mean + dtr / 2).round(1),
        "temperature_2m_min": (t_mean - dtr / 2).round(1),
        "temperature_2m_mean": t_mean.round(1),
        "apparent_temperature_max": (t_mean + dtr / 2 + app_offset).round(1),
        "apparent_temperature_min": (t_mean - dtr / 2 + app_offset).round(1),
        "apparent_temperature_mean": (t_mean + app_offset).round(1),
        "shortwave_radiation_sum": rad.round(2), "precipitation_sum": precip,
        "precipitation_hours": np.where(precip > 0, np.clip(precip / 2.2 + 1, 1, 24), 0).round(1),
        "windspeed_10m_max": wspd.round(1), "windgusts_10m_max": (wspd * 1.5).round(1),
        "winddirection_10m_dominant": wdir.round(0),
        "et0_fao_evapotranspiration": et0.round(2),
        "latitude": lat, "longitude": lon, "elevation": elev,
    })


def main(out: str, start: str = "2010-01-01", end: str = "2022-12-31", seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    frames = [simulate_city(rng, *c, dates) for c in CITIES]
    df = pd.concat(frames, ignore_index=True)
    # inject realistic missingness in radiation/ET (~1.2%) per the blueprint
    for col in ("shortwave_radiation_sum", "et0_fao_evapotranspiration"):
        mask = rng.random(len(df)) < 0.012
        df.loc[mask, col] = np.nan
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"SURROGATE dataset written: {out}  shape={df.shape}  "
          f"rain-day share={float((df['precipitation_sum'] > 1.0).mean()):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/SriLanka_Weather_Dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.out, seed=args.seed)
