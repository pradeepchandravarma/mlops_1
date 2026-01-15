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


class LLMQueryRequest(BaseModel):
    question: str


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


@app.post("/llm/query")
def llm_query(req: LLMQueryRequest):
    """
    NL -> SQL (SELECT only) -> run on RDS -> return rows + short answer
    """
    try:
        # Lazy import so the API can boot even if LLM module packaging isn't ready yet
        try:
            from llm_service import handle_llm_query
        except ModuleNotFoundError:
            # fallback if files are copied directly into /app without src as a package
            from llm_service import handle_llm_query

        result = handle_llm_query(req.question)
        return {
            "question": req.question,
            "sql": result["sql"],
            "rows": result["rows"],
            "answer": result["answer"],
        }
    except Exception as e:
        return {
            "question": req.question,
            "error": str(e),
            "sql": None,
            "rows": [],
            "answer": None,
        }

