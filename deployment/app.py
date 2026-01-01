from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---- Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

FEATURE_COLUMNS = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]


# ---- API schema ----
class PredictRequest(BaseModel):
    # keep it flexible but structured
    features: Dict[str, Any] = Field(..., description="Feature dict for one row prediction")


class PredictResponse(BaseModel):
    prediction: float


# ---- App ----
app = FastAPI(title="Student Performance Predictor", version="0.1.0")

_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                f"Run training first to create it, or set MODEL_PATH env var."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    # If model loads, we are healthy
    try:
        load_model()
        return {"status": "ok", "model_path": MODEL_PATH}
    except Exception as e:
        return {"status": "error", "detail": str(e), "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = load_model()

        # Build a single-row dataframe in the expected column order
        row = {col: req.features.get(col, None) for col in FEATURE_COLUMNS}
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # Basic guard: missing values
        missing = [c for c in FEATURE_COLUMNS if X.loc[0, c] is None]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required features: {missing}",
            )

        pred = float(model.predict(X)[0])
        return PredictResponse(prediction=pred)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
