from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Student Performance Predictor")

# load the trained pipeline (preprocess + model)
MODEL_PATH = "models/model.joblib"
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    # we accept a dict of feature_name -> value
    features: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Convert incoming features into a 1-row dataframe
    X = pd.DataFrame([req.features])
    pred = model.predict(X)[0]
    return {"prediction": float(pred)}
