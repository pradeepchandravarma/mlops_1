
import requests
import os
import streamlit as st

#API_URL = "http://127.0.0.1:8000/predict"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


st.title("Student Performance Predictor (Lasso)")

hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=7.0, step=1.0)
previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=1.0)
papers = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=50.0, value=2.0, step=1.0)

if st.button("Predict"):
    payload = {
        "Hours Studied": hours_studied,
        "Previous Scores": previous_scores,
        "Extracurricular Activities": extracurricular,
        "Sleep Hours": sleep_hours,
        "Sample Question Papers Practiced": papers,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        pred = resp.json()["prediction"]
        st.success(f"Predicted Performance Index: {pred:.2f}")
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        st.info("Make sure FastAPI is running: uvicorn src.api:app --reload")
