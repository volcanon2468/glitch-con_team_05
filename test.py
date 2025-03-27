import joblib
import numpy as np
import pandas as pd

from log_to_es import log_alert

# Load trained models, scaler, and PCA
model_data = joblib.load("ensemble_model.pkl")
models = model_data["models"]
feature_names = model_data["feature_names"]
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")

# Display feature count and names for verification
print(f"Total Features: {len(feature_names)}")
print("Feature Names:", feature_names)

# Generate a test sample with the correct number of features
test_features = np.random.rand(len(feature_names))  # Random values for testing
test_df = pd.DataFrame([test_features], columns=feature_names)

# Function to make ensemble predictions
def ensemble_predict(models, X):
    predictions = np.array([model.predict(X) for model in models])
    return np.where(np.sum(predictions, axis=0) < 0, -1, 1)  # Majority vote

# Apply MinMaxScaler & PCA before making predictions
test_scaled = scaler.transform(test_df)
test_pca = pca.transform(test_scaled)

# Make prediction
prediction = ensemble_predict(models, test_pca)
print("Anomaly Detected!" if prediction[0] == -1 else "Normal Behavior")

# Log anomaly if detected
if prediction[0] == -1:
    log_alert("192.168.1.2", -0.75, prediction.tolist())
    """unable to perform this code due to some errors"""