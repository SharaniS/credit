import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the credit card transaction lookup data
data = pd.read_csv("Credit_card_pred.csv")
data["Credit card number"] = data["Credit card number"].astype(str)

st.title("💳 Credit Card Fraud Detection")

card_number = st.text_input("Enter Credit Card Number:")

if st.button("Check Fraud Status"):
    record = data[data["Credit card number"] == card_number]

    if record.empty:
        st.warning("❗ Credit card number not found.")
    else:
        # Select only the model input features
        input_data = record[["V1", "V2", "V3", "V4", "V5", "Amount"]].values
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")
