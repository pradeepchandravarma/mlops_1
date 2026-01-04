import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/model.joblib"))
READY_PATH = Path(os.getenv("MODEL_READY_PATH", "/app/models/model.ready"))
WAIT_SECONDS = int(os.getenv("MODEL_WAIT_SECONDS", "240"))
POLL_SECONDS = float(os.getenv("MODEL_POLL_SECONDS", "2"))


def load_model_with_wait():
    deadline = time.time() + WAIT_SECONDS
    last_err = None

    while time.time() < deadline:
        if READY_PATH.exists() and MODEL_PATH.exists():
            # avoid half-written/corrupt file
            if MODEL_PATH.stat().st_size > 10_000:
                try:
                    return joblib.load(MODEL_PATH)
                except Exception as e:
                    last_err = e
        time.sleep(POLL_SECONDS)

    raise RuntimeError(
        f"Model not ready after {WAIT_SECONDS}s | "
        f"MODEL_PATH={MODEL_PATH} READY_PATH={READY_PATH} last_err={last_err}"
    )


app = FastAPI(title="Student Performance Predictor")
model = load_model_with_wait()

EXPECTED_FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]


class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24)
    previous_scores: float = Field(..., ge=0, le=100)
    extracurricular_activities: str = Field(..., pattern="^(Yes|No)$")
    sleep_hours: float = Field(..., ge=0, le=24)
    sample_question_papers_practiced: float = Field(..., ge=0, le=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        X = pd.DataFrame([{
            "Hours Studied": req.hours_studied,
            "Previous Scores": req.previous_scores,
            "Extracurricular Activities": req.extracurricular_activities,
            "Sleep Hours": req.sleep_hours,
            "Sample Question Papers Practiced": req.sample_question_papers_practiced,
        }])[EXPECTED_FEATURES]

        pred = model.predict(X)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")