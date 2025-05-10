import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv('Credit_card_pred.csv')

# Define the features used by the model
feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Extract features
X = df[feature_columns]

# Load your trained model
model = joblib.load('credit_model.pkl')

# Make predictions
predictions = model.predict(X)

# Add predictions to the DataFrame
results_df = X_test.copy()
results_df['Prediction'] = predictions  # ✅ Safe way
df['Prediction_Label'] = df['Prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

# Show the first 10 results
df[['Time', 'Amount', 'Prediction_Label']].head(10)
