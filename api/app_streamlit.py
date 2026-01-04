import os
import requests
import streamlit as st

# ---------------------------
# Configuration
# ---------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Student Performance Predictor")
st.write("Enter student details to predict the performance index.")

# ---------------------------
# Input Form
# ---------------------------
with st.form("prediction_form"):
    hours_studied = st.number_input("Hours Studied", 0.0, 24.0, 6.0)
    previous_scores = st.number_input("Previous Scores", 0.0, 100.0, 75.0)
    extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    sample_question_papers_practiced = st.number_input(
        "Sample Question Papers Practiced", 0.0, 50.0, 4.0
    )

    submit = st.form_submit_button("Predict")

# ---------------------------
# Prediction
# ---------------------------
if submit:
    payload = {
        "hours_studied": hours_studied,
        "previous_scores": previous_scores,
        "extracurricular_activities": extracurricular_activities,
        "sleep_hours": sleep_hours,
        "sample_question_papers_practiced": sample_question_papers_practiced,
    }

    try:
        with st.spinner("Predicting..."):
            response = requests.post(API_URL, json=payload, timeout=5)
            response.raise_for_status()

            prediction = response.json()["prediction"]
            st.success(f"🎯 Predicted Performance Index: **{prediction:.2f}**")

    except requests.exceptions.RequestException as e:
        st.error(f"❌ API error: {e}")
