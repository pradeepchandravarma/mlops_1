import os
import streamlit as st
import requests

# ✅ Default to ECS Service Discovery (Cloud Map) inside the VPC
DEFAULT_API_BASE_URL = "http://api.mary-api:8000"

# Optional override (only if you want it)
API_BASE_URL = os.getenv("API_URL", DEFAULT_API_BASE_URL).rstrip("/")

PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

st.write("Calling API at:", PREDICT_URL)

st.title("Maryam Student Performance Predictor (UI)")

# (Optional) quick connectivity check button
with st.expander("API Connectivity Check", expanded=False):
    if st.button("Check API health"):
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            st.success(f"Health OK ({r.status_code}): {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            st.error(f"Health check failed: {e}")

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
        data = r.json()

        # Safer parsing
        pred = data.get("prediction", None)
        if pred is None:
            st.error(f"Unexpected response JSON: {data}")
        else:
            st.success(f"Predicted Performance Index: {float(pred):.2f}")

    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
