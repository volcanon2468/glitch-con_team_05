# glitch-con_team_05
problem statement:"AI-Powered Cyber Threat Detection System"

# Files in the Repository:

app.py: Main application script for running the project.

train.py: Script for training the machine learning models.

test.py & test_primary.py: Scripts for testing model performance.

pca_process.py: Handles PCA transformation on the dataset.

check_model_limits.py: Validates model constraints and performance limits.

log_to_es.py: Implements logging functionality to store outputs in Elasticsearch.

ensemble_model.pkl: Pre-trained ensemble model.

pca_model.pkl: Saved PCA transformation model.

scaler.pkl: Saved scaler model for data normalization

# What I Tried to Do
Implemented a machine learning pipeline with PCA and scaling.

Used ensemble methods to improve prediction accuracy.

Added logging support via Elasticsearch for tracking performance.

Validated model constraints to ensure robustness.

And to use this model for real time use

# Shortcomings and Challenges

PCA Impact on Model Performance: The dimensionality reduction did not always improve accuracy as expected.

Logging Issues: Encountered difficulties in efficiently integrating Elasticsearch logging.

Model Generalization: The ensemble model sometimes overfit to specific datasets.

Scalability: Processing large datasets led to performance bottlenecks.
