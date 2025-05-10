import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv('Credit_card_pred.csv')

# Clean column names to avoid issues with extra spaces or case mismatches
df.columns = df.columns.str.strip().str.lower()  # Strip spaces and make lowercase

# Define the feature columns as per the model's training data
feature_columns = ['time'] + [f'v{i}' for i in range(1, 29)] + ['amount']  # Adjust according to your dataset and training

# Check if all feature columns are present in the dataframe
missing_columns = [col for col in feature_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Extract the features (X) from the dataframe
X = df[feature_columns]

# Handle missing values if there are any (you may need to replace with 0 or other strategies)
X = X.fillna(0)  # You can change this to another strategy depending on your model's needs

# Load your trained model
model = joblib.load('credit_model (3).pkl')  # Ensure the path to the model is correct

# Make predictions
predictions = model.predict(X)

# Add predictions to the original dataframe
df['prediction'] = predictions

# Map predictions to 'Fraud' and 'Not Fraud'
df['prediction_label'] = df['prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

# Show the first 10 rows (Time, Amount, and Prediction Label)
results = df[['time', 'amount', 'prediction_label']].head(10)

# Display the results
print(results)
