from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI(title="Student Performance Predictor")

# Load artifacts once
model = tf.keras.models.load_model("artifacts/model.h5",compile=False)
scaler = joblib.load("artifacts/scaler.joblib")


# ---------------------------
# Request Schema
# ---------------------------
class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24)
    previous_scores: float = Field(..., ge=0, le=100)
    extracurricular_activities: str = Field(..., pattern="^(Yes|No)$")
    sleep_hours: float = Field(..., ge=0, le=24)
    sample_question_papers_practiced: float = Field(..., ge=0, le=100)


# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        input_data = np.array([[
            req.hours_studied,
            req.previous_scores,
            1 if req.extracurricular_activities == "Yes" else 0,
            req.sleep_hours,
            req.sample_question_papers_practiced
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0][0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
