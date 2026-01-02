from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Student Performance Predictor")

MODEL_PATH = Path("models/model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found: {MODEL_PATH.resolve()}")

model = joblib.load(MODEL_PATH)

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
        }])

        # Force correct schema + order
        X = X[EXPECTED_FEATURES]

        pred = model.predict(X)[0]
        return {"prediction": float(pred)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")