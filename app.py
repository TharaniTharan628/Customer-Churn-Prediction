import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn")

st.divider()

# User inputs
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

if st.button("Predict Churn"):
    prediction = model.predict([[tenure, monthly_charges]])

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
