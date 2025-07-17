import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = r"C:\MLCourse\My_Exercises\Autonomous System\Data\filtered_sensor_data.csv"  # Update with actual filename

def load_data(path=DATA_PATH):
    """
    Load sensor data from csv file.
    """
    df = pd.read_csv(path)
    return df

def normalize_data(df, method="minmax"):
    """
    Normalize sensor values using MinMax or Z-score.
    """
    if method == "minmax":
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
    elif method == "zscore":
        scaled = (df - df.mean()) / df.std()
    else:
        raise ValueError("Unsupported normalization method")
    return scaled

def segment_windows(data, window_size=50, step_size=10):
    """
    Segment time-series data using sliding window.
    Returns shape: [num_windows, window_size, num_features]
    """
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start+window_size]
        windows.append(window)
    return np.array(windows)

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded data shape: {df.shape}")
    data_norm = normalize_data(df)
    windows = segment_windows(data_norm)
    print(f"Windowed data shape: {windows.shape}")