import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv('Credit_card_pred.csv')

# Clean column names to avoid issues with extra spaces
df.columns = df.columns.str.strip().str.lower()  # Adjust column name formatting

# Define the features used by the model
feature_columns = ['time'] + [f'v{i}' for i in range(1, 29)] + ['amount']  # Adjust to match your actual column names

# Extract features
X = df[feature_columns]

# Handle missing values (if any)
X = X.fillna(0)  # Replace missing values with 0 (or use another appropriate strategy)

# Load your trained model
model = joblib.load('credit_model (3).pkl')

# Make predictions
predictions = model.predict(X)

# Add predictions to the DataFrame
df['prediction'] = predictions  # Add prediction results to the original DataFrame

# Map predictions to labels (0 -> 'Not Fraud', 1 -> 'Fraud')
df['prediction_label'] = df['prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

# Show the first 10 results (Time, Amount, and Prediction Label)
results = df[['time', 'amount', 'prediction_label']].head(10)

# Display the results
print(results)
