"""FastAPI serving layer: real-time /predict and bulk /predict_batch.

Models are loaded once at startup (lifespan); inference itself is CPU-bound
and fast (<10 ms), so endpoints are async with the prediction call kept
inline. Request/response pairs are appended to a JSONL prediction log that
the drift monitor (app/monitoring.py) consumes.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchRequest,
    BatchResponse,
    DayFeatures,
    Health,
    Prediction,
)

MODEL_DIR = Path("models")
PREDICTION_LOG = Path("reports/prediction_log.jsonl")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    clf = CatBoostClassifier()
    clf.load_model(str(MODEL_DIR / "catboost_rain.cbm"))
    reg = CatBoostRegressor()
    reg.load_model(str(MODEL_DIR / "catboost_precip_amount.cbm"))
    spec = joblib.load(MODEL_DIR / "feature_spec.joblib")
    thr_file = MODEL_DIR / "decision_threshold.json"
    threshold = 0.5
    if thr_file.exists():
        threshold = json.loads(thr_file.read_text())["threshold"]
    STATE.update(
        clf=clf, reg=reg, feat_cols=spec["feat_cols"], threshold=threshold,
        version=f"catboost-{int(clf.tree_count_)}trees",
    )
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    yield
    STATE.clear()


app = FastAPI(
    title="Sri Lanka Rain Early-Warning API",
    description="Monsoon-aware next-day rain probability and expected "
                "precipitation for 30 Sri Lankan cities.",
    version="1.0.0",
    lifespan=lifespan,
)


def _predict_frame(df: pd.DataFrame) -> list[Prediction]:
    feat_cols = STATE["feat_cols"]
    # imputation indicator columns default to 0 (client data is observed)
    for col in feat_cols:
        if col.endswith("_imputed") and col not in df.columns:
            df[col] = 0
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise HTTPException(422, detail=f"Missing features: {missing}")
    X = df[feat_cols]
    proba = STATE["clf"].predict_proba(X)[:, 1]
    amount = np.expm1(STATE["reg"].predict(X))
    expected = np.clip(proba * amount, 0, None)
    thr = STATE["threshold"]
    out = [
        Prediction(
            city=str(df.iloc[i]["city"]),
            rain_tomorrow_prob=round(float(proba[i]), 4),
            alert=bool(proba[i] >= thr),
            expected_precip_mm=round(float(expected[i]), 2),
            threshold=thr,
        )
        for i in range(len(df))
    ]
    with open(PREDICTION_LOG, "a", encoding="utf-8") as fh:
        for i, p in enumerate(out):
            fh.write(json.dumps({"ts": time.time(),
                                 **df.iloc[i][feat_cols].to_dict(),
                                 "proba": p.rain_tomorrow_prob}) + "\n")
    return out


@app.get("/health", response_model=Health)
async def health() -> Health:
    return Health(status="ok", model_loaded="clf" in STATE,
                  model_version=STATE.get("version", "unloaded"))


@app.post("/predict", response_model=Prediction)
async def predict(x: DayFeatures) -> Prediction:
    df = pd.DataFrame([x.model_dump()])
    return _predict_frame(df)[0]


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest) -> BatchResponse:
    if not req.observations:
        raise HTTPException(422, detail="Empty batch")
    df = pd.DataFrame([o.model_dump() for o in req.observations])
    return BatchResponse(predictions=_predict_frame(df),
                         model_version=STATE["version"])
