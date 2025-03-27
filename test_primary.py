import requests
import numpy as np

# Generate 78 random feature values
random_features = np.random.rand(78).tolist()  # Convert to list format

# Use feature names as strings ("0", "1", ..., "77")
feature_names = [str(i) for i in range(78)]
payload = {feature_names[i]: float(random_features[i]) for i in range(78)}

# Send request
url = "http://127.0.0.1:5001/predict"
response = requests.post(url, json=payload)

# Print API response
print(response.json())