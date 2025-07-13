import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# Load model, scaler, and column order
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
with open("columns.json", "r") as f:
    column_order = json.load(f)

st.title("💊 Medical Insurance Charges Predictor")

st.markdown("Enter your details below:")

# Input fields
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker?", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "southeast", "southwest", "northwest"])

# Prepare input for model
input_data = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0
}

# Add missing dummy columns
for col in column_order:
    if col not in input_data:
        input_data[col] = 0

# Reorder and scale
X_input = pd.DataFrame([input_data])[column_order]
X_scaled = scaler.transform(X_input)

# Predict
if st.button("Predict Charges"):
    log_pred = model.predict(X_scaled)[0]
    charge_pred = np.expm1(log_pred)
    st.success(f"💰 Estimated Insurance Charges: ₹{charge_pred:,.2f}")