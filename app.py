import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_model.pkl")
expected_features = model.feature_names_in_

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details:")

# Dynamically create input fields based on the model's expected features
user_input = {}
for feature in expected_features:
    user_input[feature] = st.number_input(feature)

# Predict when button is clicked
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([user_input])  # match expected columns
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Alert! This transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ This transaction is predicted to be LEGITIMATE.")
