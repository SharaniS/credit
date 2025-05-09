import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_model.pkl")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Credit_card_pred.csv")

df = load_data()

# Streamlit UI
st.title("Credit Card Fraud Detection")
card_input = st.text_input("Enter Credit Card Number:")

if card_input:
    try:
        # Convert to int if needed (adjust based on your dataset's format)
        card_input = int(card_input)

        # Find matching row
        matching_row = df[df['Credit card number'] == card_input]

        if not matching_row.empty:
            # Drop non-feature columns
            features = matching_row.drop(columns=['Credit card number', 'Time', 'Class'])

            # Predict
            prediction = model.predict(features)[0]

            if prediction == 1:
                st.error("⚠️ This transaction is predicted to be FRAUDULENT.")
            else:
                st.success("✅ This transaction is predicted to be NOT FRAUDULENT.")
        else:
            st.warning("No matching credit card number found in the dataset.")

    except ValueError:
        st.error("Invalid credit card number format.")
