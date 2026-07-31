"""Pydantic v2 schemas for the early-warning REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DayFeatures(BaseModel):
    """One city-day observation with engineered features (day t) used to
    predict rain on day t+1. Field names match the training feature spec."""

    city: str = Field(..., examples=["Colombo"])
    monsoon_phase: str = Field(..., pattern="^(SW|NE|inter1|inter2)$")
    climate_zone: str = Field(..., pattern="^(wet_zone|dry_zone|hill_country)$")
    temperature_2m_max: float
    temperature_2m_min: float
    temperature_2m_mean: float
    apparent_temperature_max: float
    apparent_temperature_min: float
    apparent_temperature_mean: float
    shortwave_radiation_sum: float = Field(..., ge=0)
    precipitation_sum: float = Field(..., ge=0)
    precipitation_hours: float = Field(..., ge=0, le=24)
    windspeed_10m_max: float = Field(..., ge=0)
    windgusts_10m_max: float = Field(..., ge=0)
    winddirection_10m_dominant: float = Field(..., ge=0, lt=360)
    et0_fao_evapotranspiration: float = Field(..., ge=0)
    latitude: float
    longitude: float
    elevation: float
    precip_anom_7d: float
    precip_anom_15d: float
    precip_anom_30d: float
    heat_index_f: float
    rh_proxy: float = Field(..., ge=0, le=100)
    month_sin: float = Field(..., ge=-1, le=1)
    month_cos: float = Field(..., ge=-1, le=1)
    dtr: float
    wind_u: float
    wind_v: float
    precip_to_et: float
    precip_lag1: float = Field(..., ge=0)
    precip_lag2: float = Field(..., ge=0)
    precip_lag3: float = Field(..., ge=0)
    precip_lag7: float = Field(..., ge=0)
    rain_today: int = Field(..., ge=0, le=1)
    rain_yesterday: int = Field(..., ge=0, le=1)


class Prediction(BaseModel):
    city: str
    rain_tomorrow_prob: float
    alert: bool
    expected_precip_mm: float
    threshold: float


class BatchRequest(BaseModel):
    observations: list[DayFeatures]


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
