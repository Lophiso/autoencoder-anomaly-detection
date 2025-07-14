import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        # Decoder
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        _, (h_n, _) = self.encoder(x)
        # Repeat embedding for each timestep in sequence
        repeated_emb = h_n.repeat(self.seq_len, 1, 1).permute(1,0,2)
        decoded, _ = self.decoder(repeated_emb)
        return decoded

if __name__ == "__main__":
    # Test forward pass
    seq_len, n_features = 50, 6
    batch_size = 8
    model = LSTMAutoencoder(seq_len, n_features)
    x = torch.randn(batch_size, seq_len, n_features)
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)