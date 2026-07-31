"""Unit tests: leakage-safety of the CV split, feature math, API schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import _heat_index_f, engineer_features
from src.train import PanelTimeSeriesSplit
from src.utils import assign_monsoon_phase, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/config.yaml")


@pytest.fixture(scope="module")
def toy_panel():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    frames = []
    for city, elev in [("A", 10), ("B", 800)]:
        frames.append(pd.DataFrame({
            "time": dates, "city": city,
            "temperature_2m_max": rng.normal(30, 2, 400),
            "temperature_2m_min": rng.normal(23, 2, 400),
            "temperature_2m_mean": rng.normal(26.5, 2, 400),
            "apparent_temperature_mean": rng.normal(28, 2, 400),
            "precipitation_sum": rng.gamma(0.5, 6, 400) * (rng.random(400) < 0.4),
            "windspeed_10m_max": rng.gamma(4, 3, 400),
            "winddirection_10m_dominant": rng.uniform(0, 360, 400),
            "et0_fao_evapotranspiration": rng.uniform(2, 6, 400),
            "latitude": 6.9, "longitude": 79.9, "elevation": elev,
        }))
    return pd.concat(frames, ignore_index=True)


class TestPanelTimeSeriesSplit:
    def test_no_temporal_leakage(self, toy_panel):
        """Every training date must strictly precede every validation date."""
        cv = PanelTimeSeriesSplit(n_splits=5, embargo_days=1)
        dates = toy_panel["time"]
        folds = list(cv.split(dates))
        assert len(folds) == 5
        for tr, va in folds:
            assert dates.iloc[tr].max() < dates.iloc[va].min()

    def test_all_cities_in_both_sides(self, toy_panel):
        cv = PanelTimeSeriesSplit(n_splits=5, embargo_days=1)
        for tr, va in cv.split(toy_panel["time"]):
            assert set(toy_panel.iloc[tr]["city"]) == {"A", "B"}
            assert set(toy_panel.iloc[va]["city"]) == {"A", "B"}

    def test_expanding_window(self, toy_panel):
        cv = PanelTimeSeriesSplit(n_splits=5, embargo_days=1)
        sizes = [len(tr) for tr, _ in cv.split(toy_panel["time"])]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


class TestFeatures:
    def test_target_is_next_day(self, toy_panel, cfg):
        out = engineer_features(toy_panel.copy(), cfg)
        g = out[out["city"] == "A"].sort_values("time")
        # precip_tomorrow at row t equals precipitation_sum at row t+1
        assert np.allclose(
            g["precip_tomorrow"].values[:-1],
            g["precipitation_sum"].values[1:], atol=1e-9)

    def test_anomaly_clipped(self, toy_panel, cfg):
        out = engineer_features(toy_panel.copy(), cfg)
        clip = cfg["features"]["anomaly_clip"]
        for w in cfg["features"]["anomaly_windows"]:
            assert out[f"precip_anom_{w}d"].abs().max() <= clip + 1e-9

    def test_wind_vector_magnitude(self, toy_panel, cfg):
        out = engineer_features(toy_panel.copy(), cfg)
        mag = np.hypot(out["wind_u"], out["wind_v"])
        assert np.allclose(mag, out["windspeed_10m_max"], atol=1e-6)

    def test_heat_index_rh_bounds(self):
        t = pd.Series([25.0, 32.0, 38.0])
        a = pd.Series([26.0, 36.0, 30.0])
        _, rh = _heat_index_f(t, a)
        assert rh.between(20, 100).all()

    def test_monsoon_phase(self):
        s = pd.Series(pd.to_datetime(["2022-06-15", "2022-01-15",
                                      "2022-03-15", "2022-10-15"]))
        assert list(assign_monsoon_phase(s)) == ["SW", "NE", "inter1", "inter2"]


class TestAPISchemas:
    def test_day_features_validation(self):
        from app.schemas import DayFeatures

        base = dict(
            city="Colombo", monsoon_phase="SW", climate_zone="wet_zone",
            temperature_2m_max=31.0, temperature_2m_min=25.0,
            temperature_2m_mean=28.0, apparent_temperature_max=35.0,
            apparent_temperature_min=27.0, apparent_temperature_mean=30.0,
            shortwave_radiation_sum=18.0, precipitation_sum=4.2,
            precipitation_hours=3.0, windspeed_10m_max=20.0,
            windgusts_10m_max=30.0, winddirection_10m_dominant=225.0,
            et0_fao_evapotranspiration=4.0, latitude=6.93, longitude=79.85,
            elevation=7.0, precip_anom_7d=0.5, precip_anom_15d=0.2,
            precip_anom_30d=0.1, heat_index_f=88.0, rh_proxy=80.0,
            month_sin=0.0, month_cos=-1.0, dtr=6.0, wind_u=-14.1,
            wind_v=-14.1, precip_to_et=1.05, precip_lag1=2.0,
            precip_lag2=0.0, precip_lag3=1.0, precip_lag7=0.0,
            rain_today=1, rain_yesterday=1,
        )
        assert DayFeatures(**base).city == "Colombo"
        with pytest.raises(Exception):
            DayFeatures(**{**base, "monsoon_phase": "monsoonX"})
        with pytest.raises(Exception):
            DayFeatures(**{**base, "precipitation_sum": -1.0})
