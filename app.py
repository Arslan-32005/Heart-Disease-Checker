import streamlit as st
import joblib
import numpy as np

model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Risk Checker")
st.write("Enter your details — model will tell you the risk of heart disease.")

age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0)
slope = st.selectbox("ST Slope", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
ca = st.selectbox("Major Vessels Colored", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thal", options=[0, 1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

if st.button("Check Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]
    
    if prediction[0] == 1:
        st.error(f"⚠️ Heart Disease Risk Detected — Confidence: {probability[1]*100:.1f}%")
        st.write("Please consult a doctor.")
    else:
        st.success(f"✅ Low Risk — Confidence: {probability[0]*100:.1f}%")
        st.write("You seem to be in good health — keep up the good work.")