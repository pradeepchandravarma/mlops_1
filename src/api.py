from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

from src.llm_service import handle_llm_query  # LLM endpoint support

app = FastAPI(title="Student Performance Predictor")

MODEL_PATH = "models/student_performance_sgd.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]


class PredictRequest(BaseModel):
    features: dict


class LLMQueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    X = pd.DataFrame([req.features])
    if "Extracurricular Activities" in X.columns:
        X["Extracurricular Activities"] = X["Extracurricular Activities"].replace({"Yes": 1, "No": 0})
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return {"prediction": float(pred)}


@app.post("/llm/query")
def llm_query(req: LLMQueryRequest):
    """
    NL -> SQL (SELECT only) -> run on RDS -> return rows + short answer
    """
    try:
        result = handle_llm_query(req.question)
        return {
            "question": req.question,
            "sql": result["sql"],
            "rows": result["rows"],
            "answer": result["answer"],
        }
    except RuntimeError as e:
        # usually config missing (OPENAI key / DB env / table missing)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        # usually unsafe SQL or invalid user request
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

