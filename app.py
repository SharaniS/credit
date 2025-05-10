import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("credit_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection (Static Output)")

# Predefined credit card numbers and example features (features are placeholders)
card_numbers = [
    ("6310836524182291", [0.1, -1.2, 0.5, 1.0, -0.7, 110.0]),  # Expected: False
    ("7674734919115658", [1.5, -0.8, 0.9, -1.1, 0.3, 300.0]),  # Expected: True
]

# Display predictions
for number, features in card_numbers:
    input_array = np.array([features])
    prediction = model.predict(input_array)[0]
    st.write(f"**Credit Card Number:** `{number}` — **Fraudulent:** {bool(prediction)}")
    st.markdown("---")
