import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Student Performance Predictor (UI)")

hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=6.0)
prev = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0)
extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
papers = st.number_input("Sample Question Papers Practiced", min_value=0.0, max_value=50.0, value=4.0)

if st.button("Predict"):
    payload = {
        "hours_studied": hours,
        "previous_scores": prev,
        "extracurricular_activities": extra,
        "sleep_hours": sleep,
        "sample_question_papers_practiced": papers,
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            pred = r.json()["prediction"]
            st.success(f"Predicted Performance Index: {pred:.2f}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")