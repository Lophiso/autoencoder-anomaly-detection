import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LSTMAutoencoder
from preprocessing import load_data, normalize_data, segment_windows

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load and preprocess data
    df = load_data()
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    data_norm = normalize_data(df)
    windows = segment_windows(data_norm, window_size=5, step_size=1)  # Small window for small sample

    X = torch.tensor(windows, dtype=torch.float32).to(DEVICE)

    # Model params
    seq_len = X.shape[1]
    n_features = X.shape[2]
    embedding_dim = 64

    # Load model
    model = LSTMAutoencoder(seq_len, n_features, embedding_dim).to(DEVICE)
    model.load_state_dict(torch.load("../data/lstm_autoencoder.pth"))
    model.eval()

    # Reconstruction error for each window
    with torch.no_grad():
        reconstructed = model(X)
        mse = torch.mean((X - reconstructed) ** 2, dim=(1,2)).cpu().numpy()
    
    # Visualize reconstruction error
    plt.plot(mse, label='Reconstruction Error')
    plt.xlabel('Window')
    plt.ylabel('MSE')
    plt.title('Reconstruction Error (Anomaly Score)')
    plt.legend()
    plt.show()

    # Flag anomalies (simple threshold for demo)
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold
    print(f"Anomalies at windows: {np.where(anomalies)[0]}")

if __name__ == "__main__":
    main()