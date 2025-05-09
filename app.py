import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("credit_model.pkl")

# Load dataset
data = pd.read_csv("Credit_card_pred.csv")

# Title
st.title("Credit Card Fraud Detection")

# Option to display the dataset
if st.checkbox("Show sample dataset"):
    st.write(data.head())

# Input for Credit Card Number (ID)
card_number = st.text_input("Enter Credit Card Number:")

# Prediction logic
if st.button("Predict Fraud"):
    if card_number == "":
        st.warning("Please enter a credit card number.")
    elif card_number not in data["ID"].astype(str).values:
        st.error("Credit card number not found in the dataset.")
    else:
        # Locate the record by ID
        record = data[data["ID"].astype(str) == card_number]
        features = record.drop(columns=["ID", "Class"])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Display result
        if prediction == 1:
            st.error(f"⚠️ FRAUDULENT TRANSACTION detected with probability {probability:.2f}")
        else:
            st.success(f"✅ Transaction appears legitimate with probability {1 - probability:.2f}")
