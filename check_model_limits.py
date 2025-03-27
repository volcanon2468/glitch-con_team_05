import joblib
import numpy as np

# Load the saved model
model_data = joblib.load("ensemble_model.pkl")
models = model_data["models"]  # List of Isolation Forest models
feature_names = model_data["feature_names"]

# Function to print model limits
def print_model_limits(models):
    for i, model in enumerate(models):
        print(f"\n=== Model {i+1} ===")
        if hasattr(model, "offset_"):
            print(f"Threshold (offset_): {model.offset_}")
        if hasattr(model, "threshold_"):
            print(f"Anomaly Score Threshold (threshold_): {model.threshold_}")
        if hasattr(model, "decision_function"):
            print("Decision function available: Yes")
        
        # Extracting threshold values safely
        feature_thresholds = []
        for tree in model.estimators_:
            if hasattr(tree, "tree_") and hasattr(tree.tree_, "threshold"):
                tree_thresholds = tree.tree_.threshold
                tree_thresholds = tree_thresholds[tree_thresholds != -2]  # Exclude unused nodes (-2 is a default value)
                feature_thresholds.extend(tree_thresholds)

        if feature_thresholds:
            feature_thresholds = np.array(feature_thresholds)
            print(f"Feature Thresholds (min, max): {feature_thresholds.min()}, {feature_thresholds.max()}")

# Call the function
print_model_limits(models)