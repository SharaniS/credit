import streamlit as st
import pandas as pd
import joblib

# Load your data and model
df = pd.read_csv('Credit_card_pred.csv')
model = joblib.load('your_model.pkl')

st.title("Credit Card Fraud Detection")

card_input = st.text_input("Enter Credit Card Number:")

if card_input:
    try:
        card_input = int(card_input)
        matching_row = df[df['Credit card number'] == card_input]

        if not matching_row.empty:
            features = matching_row.drop(['Credit card number', 'Time', 'Class'], axis=1)

            # Debug info
            st.write("Model Input Features:", features)

            prediction = model.predict(features)[0]

            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction.")
        else:
            st.error("Card number not found.")
    except ValueError:
        st.error("Invalid credit card number format.")
