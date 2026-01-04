import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL")

st.title("Student Performance Predictor")

payload = {}
payload["hours_studied"] = st.number_input("Hours Studied", 0.0, 24.0, 6.0)
payload["previous_scores"] = st.number_input("Previous Scores", 0.0, 100.0, 75.0)
payload["extracurricular_activities"] = st.selectbox("Activities", ["Yes", "No"])
payload["sleep_hours"] = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
payload["sample_question_papers_practiced"] = st.number_input("Papers", 0.0, 50.0, 4.0)

if st.button("Predict"):
    r = requests.post(API_URL, json=payload)
    st.success(f"Prediction: {r.json()['prediction']:.2f}")