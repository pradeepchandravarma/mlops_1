from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Student Performance Predictor")

MODEL_PATH = "models/student_performance_sgd.joblib"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]

class PredictRequest(BaseModel):
    features: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = pd.DataFrame([req.features])
    X["Extracurricular Activities"] = X["Extracurricular Activities"].replace({"Yes": 1, "No": 0})
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return {"prediction": float(pred)}
