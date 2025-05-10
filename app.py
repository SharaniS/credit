import streamlit as st
import pandas as pd

# Load dataset
data = pd.read_csv("Credit_card_pred.csv")

# Clean column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Title
st.title("Credit Card Fraud Detection")

# Option to display the dataset
if st.checkbox("Show sample dataset"):
    st.write(data.head())

# Input for Credit Card Number
card_number = st.text_input("Enter Credit Card Number:")

# Search and Predict
if st.button("Check Fraud Status"):
    if card_number == "":
        st.warning("Please enter a credit card number.")
    elif card_number not in data["Credit card number"].astype(str).values:

        st.error("Credit card number not found in the dataset.")
    else:
        # Get the corresponding transaction row
        row = data[data["Credit card number"].astype(str) == card_number]

        # Get fraud label
        fraud_label = row["isFraud"].values[0]

        if fraud_label == 1:
            st.error("⚠️ FRAUDULENT TRANSACTION recorded.")
        else:
            st.success("✅ Legitimate transaction recorded.")
