import streamlit as st
import pandas as pd
import joblib

# Load the trained model

model = joblib.load("credit\_model.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's Fraudulent or Legitimate.")

# Create input fields for key features — you can expand this list as needed

V1 = st.number\_input("V1")
V2 = st.number\_input("V2")
V3 = st.number\_input("V3")
V4 = st.number\_input("V4")
V5 = st.number\_input("V5")
V6 = st.number\_input("V6")
V7 = st.number\_input("V7")
V8 = st.number\_input("V8")
V9 = st.number\_input("V9")
V10 = st.number\_input("V10")
Amount = st.number\_input("Transaction Amount")
Time = st.number\_input("Transaction Time")

# Predict button

if st.button("🔍 Predict"):
\# You must match the order and columns your model was trained on
input\_df = pd.DataFrame(\[\[V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, Amount, Time]],
columns=\['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'Amount', 'Time'])

```
prediction = model.predict(input_df)[0]

if prediction == 1:
    st.error("⚠️ Alert! This transaction is predicted to be FRAUDULENT.")
else:
    st.success("✅ This transaction is predicted to be LEGITIMATE.")  what about this code
```
