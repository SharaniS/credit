
import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below:")

# Input fields based on your model's features
V1 = st.number_input("V1")
V2 = st.number_input("V2")
V3 = st.number_input("V3")
V4 = st.number_input("V4")
V5 = st.number_input("V5")
Amount = st.number_input("Amount")

# Combine into an input array
input_data = np.array([[V1, V2, V3, V4, V5, Amount]])

# Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
