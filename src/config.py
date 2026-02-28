import os
from pathlib import Path

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path(os.getcwd())

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

DATA_PATH = DATA_DIR / "filtered_sensor_data.csv"
LABELED_DATA_PATH = DATA_DIR / "filtered_sensor_data_with_labels.csv"
MODEL_PATH = MODEL_DIR / "lstm_autoencoder.pth"
