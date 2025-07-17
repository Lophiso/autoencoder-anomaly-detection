# Autonomous System Sensor Data Anomaly Detection using Autoencoders

## Overview
This project detects anomalies in autonomous system sensor data to identify potential cybersecurity incidents. We use an LSTM Autoencoder to learn normal patterns in the data and flag irregularities using reconstruction error.

**Key Features:**
- Supports multivariate time-series sensor data (e.g., IMU, GPS, telemetry)
- Simulates cybersecurity attacks (GPS spoofing, IMU manipulation, packet drops)
- Evaluation via precision, recall, F1-score, ROC-AUC

## Dataset
- [Udacity Self-Driving Car Dataset](https://github.com/udacity/self-driving-car)
- Download instructions and links will be added in `data/README.md`.

## Pipeline
1. Data Preprocessing (normalization, windowing)
2. Model Training (LSTM Autoencoder on normal data)
3. Attack Simulation (optional, for testing robustness)
4. Anomaly Detection (threshold on reconstruction error)
5. Evaluation (confusion matrix, F1, ROC-AUC)

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/preprocessing.py

# Train model
python src/train.py

# Simulate attacks
python src/simulate_attack.py

# Evaluate results
python src/test.py
```

## Authors
- Lophiso Feleke Shomoro