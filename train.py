import joblib
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import IsolationForest

# Load all cleaned_data*.csv files
data_files = glob.glob("./clean_data/cleaned_data*.csv")
dataframes = [pd.read_csv(file) for file in data_files]

data = pd.concat(dataframes, ignore_index=True)

# Drop non-numeric columns if present
data = data.select_dtypes(include=[np.number])

# Train multiple Isolation Forest models
num_models = 3  # Number of models in the ensemble
models = [IsolationForest(n_estimators=100, contamination='auto', random_state=i).fit(data) for i in range(num_models)]

# Save the trained models
model_data = {
    "models": models,
    "feature_names": data.columns.tolist()
}

joblib.dump(model_data, "ensemble_model.pkl")
print("Ensemble model trained and saved as ensemble_model.pkl")
