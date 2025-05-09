import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("credit_model.pkl")

# Load dataset with credit card numbers
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")  # Make sure it's in the same folder as app
    return df

df = load_data()

# App title
st.title("💳 Credit Card Fraud Detection")

# Input: credit card number
card_input = st.text_input("Enter Credit Card Number:")

if card_input:
    # Try to find the row with this credit card number
    matching_row = df[df['credit_card_no'] == card_input]

    if not matching_row.empty:
        # Drop columns that are not part of the model input
        try:
            features = matching_row.drop(columns=["credit_card_no", "Class"])
        except:
            features = matching_row.iloc[:, 1:-1]  # fallback if column names differ

        # Predict
        prediction = model.predict(features)

        # Output
        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction.")

    else:
        st.warning("❗ Credit Card Number not found in dataset.")

