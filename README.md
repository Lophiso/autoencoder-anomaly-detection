# 🔍 Anomaly Detection Pipeline — LSTM Autoencoder on Multivariate Time-Series Data

## Overview

An **end-to-end data pipeline** for detecting anomalies in multivariate sensor time-series data using an LSTM Autoencoder. This project demonstrates the full ML lifecycle: **data ingestion → preprocessing → feature engineering → model training → evaluation**, with a focus on clean, reproducible Python code.

This pipeline was developed as a course project and processes approximately 5,000 rows of real sensor data through normalization, sliding-window segmentation, and deep learning-based reconstruction error analysis.

## 🏗️ Pipeline Architecture

```
Raw CSV Data → Data Ingestion (Pandas) → Normalization (MinMaxScaler)
  → Sliding Window Segmentation (NumPy) → LSTM Autoencoder Training (PyTorch)
    → Reconstruction Error Scoring → Threshold Optimization (ROC/Youden's J)
      → Anomaly Classification → Evaluation (F1, Precision, Recall, ROC-AUC)
```

## 🔧 Key Technical Skills Demonstrated

| Skill Area | Implementation Details |
|---|---|
| **Data Preprocessing** | `Pandas` for CSV ingestion, column filtering, label separation; `scikit-learn MinMaxScaler` for feature normalization |
| **Feature Engineering** | Custom sliding-window segmentation (configurable `window_size` and `step_size`) producing `[N, T, F]` tensors from flat dataframes |
| **Deep Learning** | PyTorch LSTM Autoencoder with encoder-decoder architecture, MSE reconstruction loss, Adam optimizer |
| **Evaluation & Metrics** | Confusion matrix, precision, recall, F1-score, ROC-AUC, Youden's J statistic for dynamic threshold selection |
| **Pipeline Automation** | Modular Python scripts (`preprocessing.py`, `train.py`, `test.py`) callable sequentially from CLI |

## 📂 Project Structure

```
├── src/
│   ├── preprocessing.py   # Data loading, normalization, windowing
│   ├── model.py           # LSTM Autoencoder (PyTorch nn.Module)
│   ├── train.py           # Training loop with DataLoader
│   ├── test.py            # Evaluation with sklearn metrics
│   └── simulate_attack.py # Synthetic anomaly injection
├── notebooks/
│   ├── preprocess.ipynb   # Interactive data exploration
│   ├── train.ipynb        # Training with inline outputs
│   └── test.ipynb         # Evaluation with plots
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Lophiso/autoencoder-anomaly-detection.git
cd autoencoder-anomaly-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python src/preprocessing.py   # Step 1: Preprocess raw data
python src/train.py           # Step 2: Train LSTM Autoencoder
python src/simulate_attack.py # Step 3: (Optional) Inject synthetic anomalies
python src/test.py            # Step 4: Evaluate and generate metrics
```

## 📊 Data Preprocessing Details

The preprocessing module (`src/preprocessing.py`) handles:

1. **CSV Ingestion**: Loads multivariate sensor data (16 columns, ~4,894 rows) via `pandas.read_csv()`
2. **Column Filtering**: Drops non-numeric columns (e.g., timestamps) before normalization
3. **MinMax Normalization**: Scales all features to `[0, 1]` range using `sklearn.preprocessing.MinMaxScaler` to prevent feature dominance
4. **Sliding Window Segmentation**: Converts flat time-series into overlapping windows of shape `[num_windows, 50, num_features]` with configurable step size — a technique directly applicable to **chunking text data for LLM training**

## 🤖 Model Architecture

- **Encoder**: LSTM layer (`n_features → 64`) compresses each window into a fixed-length embedding
- **Decoder**: LSTM layer (`64 → n_features`) reconstructs the original sequence
- **Anomaly Signal**: Windows with reconstruction error above a learned threshold are flagged as anomalous

## 📈 Results

- Dynamic threshold via ROC curve optimization (Youden's J statistic)
- Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Jupyter`

## 👤 Author

**Lophiso Feleke Shomoro**
M.Sc. Internet and Multimedia Engineering
```
