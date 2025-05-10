import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("credit_model.pkl")  # Make sure the file exists in the same directory

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's Fraudulent or Legitimate.")

# Create input fields for key features (adjust based on what the model expects)
V1 = st.number_input("V1")
V2 = st.number_input("V2")
V3 = st.number_input("V3")
V4 = st.number_input("V4")
V5 = st.number_input("V5")
V6 = st.number_input("V6")
V7 = st.number_input("V7")
V8 = st.number_input("V8")
V9 = st.number_input("V9")
V10 = st.number_input("V10")
Amount = st.number_input("Transaction Amount")
Time = st.number_input("Transaction Time")

# Predict button
if st.button("🔍 Predict"):
    # Create DataFrame in the order expected by the model
    input_df = pd.DataFrame(
        [[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, Amount, Time]],
        columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'Amount', 'Time']
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Alert! This transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ This transaction is predicted to be LEGITIMATE.")
