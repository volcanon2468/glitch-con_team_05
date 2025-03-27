"""import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained models, scaler, and PCA
model_data = joblib.load("ensemble_model.pkl")
models = model_data["models"]
feature_names = model_data["feature_names"]
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check if the request has features as a dictionary or a list
        if "features" in data and isinstance(data["features"], list):
            features = np.array(data["features"]).reshape(1, -1)
            features_df = pd.DataFrame(features, columns=feature_names)
        else:
            features_df = pd.DataFrame([data])  # Directly use provided feature dict
        
        # Validate input feature length
        if features_df.shape[1] != len(feature_names):
            return jsonify({"error": f"Expected {len(feature_names)} features, got {features_df.shape[1]}. Expected features: {feature_names}"})

        # Apply MinMaxScaler
        scaled_features = scaler.transform(features_df)

        # Apply PCA transformation
        pca_features = pca.transform(scaled_features)

        # Get predictions from all models
        predictions = [model.predict(pca_features)[0] for model in models]

        # Majority voting (if at least 2 models agree)
        final_prediction = "Abnormal" if predictions.count(-1) >= 2 else "Normal"

        return jsonify({"prediction": final_prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

    print("Expected feature names:", feature_names)"""
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained models
model_data = joblib.load("ensemble_model.pkl")
models = model_data["models"]
feature_names = [str(i) for i in range(78)]  # Ensure expected feature names are correct

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Debugging: Print received data
        print("Received Data:", data)

        # Ensure data is a dictionary
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected a JSON object with numerical string keys."})

        # Convert input data into DataFrame
        features_df = pd.DataFrame([data])

        # Debugging: Print received feature count
        print("Received Features Count:", len(features_df.columns))

        # Ensure the feature names match
        if set(features_df.columns) != set(feature_names):
            return jsonify({"error": f"Feature name mismatch. Expected features: {feature_names}"})

        # Convert values to float
        features_df = features_df.astype(float)

        # Get predictions from all models
        predictions = [model.predict(features_df)[0] for model in models]

        # Majority voting (if at least 2 models agree)
        final_prediction = "Abnormal" if predictions.count(-1) >= 2 else "Normal"

        return jsonify({"prediction": final_prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

"""Here we gave the data to the trained model ensemble_model.pkl in the part of the code that is not commented
The problem we were facing is that we changed the data set to a different format to get cleaned data 
to train it ;
so we tried to use pca_model.pkl to convert the data but we were facing multiple errors (primarily-some error feature name )"""