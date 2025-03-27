import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA
import joblib

# Load and merge all CSV files matching 'cleaned_data*.csv'
all_files = glob.glob("./clean_data/cleaned_data*.csv")  # Find all matching files
dataframes = [pd.read_csv(f) for f in all_files]  # Read each CSV
data = pd.concat(dataframes, ignore_index=True)  # Merge into one DataFrame

# Apply PCA (keeping 95% variance)
pca = PCA(n_components=0.95)
transformed_data = pca.fit_transform(data)

# Save PCA model
joblib.dump(pca, "pca_model.pkl")

# Save transformed dataset (optional, for verification)
pd.DataFrame(transformed_data).to_csv("pca_transformed_data.csv", index=False)

print(f"PCA Model trained on {len(data)} samples from {len(all_files)} files.")