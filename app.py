import streamlit as st
import pandas as pd

# Load dataset
data = pd.read_csv("Credit_card_pred.csv")

# Title
st.title("Credit Card Fraud Detection")

# Option to display the dataset
if st.checkbox("Show sample dataset"):
    st.write(data.head())

# Input for Credit Card Number (ID)
card_number = st.text_input("Enter Credit Card Number:")

# Search logic based on actual 'Class' column
if st.button("Check Fraud Status"):
    if card_number == "":
        st.warning("Please enter a credit card number.")
    elif card_number not in data["ID"].astype(str).values:
        st.error("Credit card number not found in the dataset.")
    else:
        row = data[data["ID"].astype(str) == card_number]
        fraud_label = row["Class"].values[0]

        if fraud_label == 1:
            st.error("⚠️ FRAUDULENT TRANSACTION recorded.")
        else:
            st.success("✅ Legitimate transaction recorded.")
