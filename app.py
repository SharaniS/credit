import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_model.pkl")

# Title
st.title("Credit Card Fraud Detection")

# Load dataset
data = pd.read_csv("Credit_card_pred.csv")

# Show dataset option
if st.checkbox("Show sample dataset"):
    st.write(data.head())

# Select a row for prediction
row_index = st.number_input("Select Row Index for Prediction", min_value=0, max_value=len(data)-1, value=0)

# Extract features (exclude the target label 'Class')
features = data.drop(columns=["Class"]).iloc[[row_index]]

# Predict
if st.button("Predict Fraud"):
    prediction = model.predict(features)[0]
    if prediction == 1:
        st.error("⚠️ Alert: This transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ This transaction is predicted to be NOT FRAUDULENT.")
