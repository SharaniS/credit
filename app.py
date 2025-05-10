import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the transaction data with credit card numbers
data = pd.read_csv("Credit_card_pred.csv")
data["Credit card number"] = data["Credit card number"].astype(str)

# Streamlit app UI
st.title("💳 Credit Card Fraud Detection")
card_number = st.text_input("Enter Credit Card Number:")

if st.button("Check Fraud Status"):
    # Filter the record for the given card number
    record = data[data["Credit card number"] == card_number]

    if record.empty:
        st.warning("❗ Credit card number not found.")
    else:
        try:
            # Extract the 32 features used for prediction
            feature_columns = ['Time'] + [f'V{i}' for i in range(1, 32)] + ['Amount']
            input_data = record[feature_columns].values

            # Predict
            prediction = model.predict(input_data)[0]

            # Display result
            if prediction == 1:
                st.error("🚨 Fraudulent Transaction Detected!")
            else:
                st.success("✅ Transaction is Legitimate.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
