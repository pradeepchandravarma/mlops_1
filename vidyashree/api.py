from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import Config

cfg = Config()
app = FastAPI(title="Student Performance Predictor", version="1.0.0")

_model = None


class StudentFeatures(BaseModel):
    Hours_Studied: float = Field(..., alias="Hours Studied")
    Previous_Scores: float = Field(..., alias="Previous Scores")
    Extracurricular_Activities: str = Field(..., alias="Extracurricular Activities")  # Yes/No
    Sleep_Hours: float = Field(..., alias="Sleep Hours")
    Sample_Question_Papers_Practiced: float = Field(..., alias="Sample Question Papers Practiced")

    class Config:
        populate_by_name = True


@app.on_event("startup")
def load_model():
    global _model
    if not Path(cfg.model_path).exists():
        raise RuntimeError(
            f"Model not found at {cfg.model_path}. Train first: python src/train.py"
        )
    _model = joblib.load(cfg.model_path)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
def predict(payload: StudentFeatures):
    # Convert to the exact column names your pipeline expects
    row = {
        "Hours Studied": payload.Hours_Studied,
        "Previous Scores": payload.Previous_Scores,
        "Extracurricular Activities": payload.Extracurricular_Activities,
        "Sleep Hours": payload.Sleep_Hours,
        "Sample Question Papers Practiced": payload.Sample_Question_Papers_Practiced,
    }
    X = pd.DataFrame([row])
    pred = float(_model.predict(X)[0])
    return {"prediction": pred}
