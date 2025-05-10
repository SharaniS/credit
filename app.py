import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("credit_model (1).pkl", "rb") as f:
    model = pickle.load(f)

# Load transaction data
data = pd.read_csv("Credit_card_pred.csv")
data["Credit card number"] = data["Credit card number"].astype(str)

# Define only the available and used features
feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")
card_number = st.text_input("Enter Credit Card Number:")

if st.button("Check Fraud Status"):
    record = data[data["Credit card number"] == card_number]

    if record.empty:
        st.warning("❗ Credit card number not found.")
    else:
        try:
            # Make sure only expected features are used
            input_data = record[feature_columns].values

            # Predict
            predictions = model.predict(input_data)

            # Display result for each transaction
            for i, prediction in enumerate(predictions):
                st.markdown(f"**Transaction {i+1}:**")
                if prediction == 1:
                    st.error("🚨 Fraudulent Transaction Detected!")
                else:
                    st.success("✅ Transaction is Legitimate.")
        except KeyError as e:
            st.error(f"Missing column(s): {e}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
