import os
import streamlit as st
import requests

# Base API URL (no /predict hardcoded)
API_BASE_URL = os.getenv("API_URL", "apicontainer:8000")

# Final predict endpoint
PREDICT_URL = f"{API_BASE_URL.rstrip('/')}/predict"

st.write("Calling API at:", PREDICT_URL)

st.title("Suganthy V1 - Student Performance Predictor (UI)")

hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=6.0)
prev = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
papers = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=50.0, value=4.0)

if st.button("Predict"):
    payload = {
        "features": {
            "Hours Studied": hours,
            "Previous Scores": prev,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleep,
            "Sample Question Papers Practiced": papers,
        }
    }

    try:
        r = requests.post(PREDICT_URL, json=payload, timeout=10)
        r.raise_for_status()
        pred = r.json()["prediction"]
        st.success(f"Predicted Performance Index: {pred:.2f}")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
