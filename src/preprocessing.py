import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

try:
    from src.config import DATA_PATH
except ImportError:
    from config import DATA_PATH

def load_data(path=DATA_PATH):
    """Load sensor data from csv file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. See data/README.md for download instructions.")
    df = pd.read_csv(path)
    return df

def normalize_data(df, method="minmax"):
    """Normalize sensor values using MinMax or Z-score."""
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    if method == "minmax":
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
    elif method == "zscore":
        scaled = (df - df.mean()) / df.std()
    else:
        raise ValueError("Unsupported normalization method")
    return scaled

def segment_windows(data, window_size=50, step_size=25):
    """
    Segment time-series data using sliding window.
    Returns shape: [num_windows, window_size, num_features]
    """
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    return np.array(windows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess sensor data")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV data file")
    args = parser.parse_args()

    df = load_data(args.data_path)
    print(f"Loaded data shape: {df.shape}")
    data_norm = normalize_data(df)
    windows = segment_windows(data_norm)
    print(f"Windowed data shape: {windows.shape}")
