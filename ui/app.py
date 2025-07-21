import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
# ✅ Corrected filename
model = joblib.load("../models/final_model.pkl")
# ✅ Make sure this file exists
scaler = joblib.load("../models/scaler.pkl")

# Streamlit App
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Fill out the form below to predict your risk of heart disease.")

# 📘 Explanation of input fields
with st.expander("ℹ️ What do the inputs mean?"):
    st.markdown("""
    - **age**: Age in years  
    - **sex**: Biological sex (1 = male, 0 = female)  
    - **cp**: Chest pain type  
        - 1 = typical angina  
        - 2 = atypical angina  
        - 3 = non-anginal pain  
        - 4 = asymptomatic  
    - **trestbps**: Resting blood pressure (in mm Hg)  
    - **chol**: Serum cholesterol (in mg/dl)  
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)  
    - **restecg**: Resting electrocardiographic results  
        - 0 = normal  
        - 1 = ST-T wave abnormality  
        - 2 = probable or definite left ventricular hypertrophy  
    - **thalach**: Maximum heart rate achieved  
    - **exang**: Exercise induced angina (1 = yes; 0 = no)  
    - **oldpeak**: ST depression induced by exercise  
    - **slope**: Slope of the peak exercise ST segment (1 = up, 2 = flat, 3 = down)  
    - **ca**: Number of major vessels colored by fluoroscopy (0–3)  
    - **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)  
    """)

# 🧾 Input Form
with st.form("heart_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox(
        "Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input(
        "ST Depression (oldpeak)", value=1.0, format="%.1f")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia", [3.0, 6.0, 7.0])

    submitted = st.form_submit_button("Predict")

# 🧠 Prediction
if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input using same scaler from training
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    st.success(f"🎯 Prediction: Heart Disease Type {prediction}")
    st.markdown("""
    **Note:**  
    - 0 = No disease  
    - 1–4 = Different heart disease levels/types (as per dataset)
    """)

# Footer
st.markdown("---")
st.caption("Developed as part of the AI/ML Summer Course Graduation Project")
