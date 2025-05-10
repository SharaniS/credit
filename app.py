import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv('Credit_card_pred.csv')

# Clean column names to ensure consistency
df.columns = df.columns.str.strip()  # Remove any leading or trailing spaces
df.columns = df.columns.str.capitalize()  # Capitalize the column names to match training

# Define the feature columns dynamically if the model provides them
try:
    model = joblib.load('credit_model (3).pkl')  # Ensure the path to the model is correct
    feature_columns = model.feature_names_in_  # Use model's feature names if available
except AttributeError:
    feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Remove unnecessary columns
columns_to_drop = ['Class', 'Predictions', 'Credit card number']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Ensure all required features are present in the DataFrame
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0  # Fill missing columns with default value

# Extract the features (X) from the dataframe
X = df[feature_columns]

# Handle missing values
X = X.fillna(0)

# Make predictions
predictions = model.predict(X)

# Add predictions to the original dataframe
df['prediction'] = predictions

# Map predictions to 'Fraud' and 'Not Fraud'
df['prediction_label'] = df['prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

# Show the first 10 rows (Time, Amount, and Prediction Label)
results = df[['Time', 'Amount', 'prediction_label']].head(10)

# Display the results
print(results)
