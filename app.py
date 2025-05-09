import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("credit_model.pkl")
  # Make sure this matches your model filename

# Load the dataset
data = pd.read_csv("Credit_card_pred.csv")

# Drop the target column (if it's included)
if 'Class' in data.columns:
    features_df = data.drop(columns=['Class'])
else:
    features_df = data.copy()

# Streamlit UI
st.title("Credit Card Fraud Detection")
card_number = st.text_input("Enter Credit Card Number:")

if card_number:
    if card_number.isdigit() and len(card_number) >= 6:  # Basic format check
        matched_row = features_df[features_df['Credit card number'] == int(card_number)]

        if not matched_row.empty:
            features = matched_row.drop(columns=['Credit card number'])
            prediction = model.predict(features)[0]
            
            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction.")
        else:
            st.warning("Credit card number not found in the dataset.")
    else:
        st.error("Invalid credit card number format.")
