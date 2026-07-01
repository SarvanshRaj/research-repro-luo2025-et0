# model.py — reimplemented from Luo et al. 2025 — SR
# GRU, LSTM, RNN, TCN, L-Transformer architectures
# paper param counts: GRU 25313, LSTM 30625, RNN 2049, TCN 1617, L-Transformer 579

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GRUModel(nn.Module):
    """GRU-based ET0 forecaster.
    Paper reports 25,313 params — we target similar.
    Architecture: GRU → FC → output
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        # take last time step
        out = gru_out[:, -1, :]
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out.squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM-based ET0 forecaster.
    Paper reports 30,625 params.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out.squeeze(-1)


class RNNModel(nn.Module):
    """Simple RNN-based ET0 forecaster.
    Paper reports 2,049 params — very small.
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = rnn_out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


class TemporalBlock(nn.Module):
    """Single temporal block for TCN — dilated causal convolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = x

        out = self.conv1(x)
        # causal: trim the future
        out = out[:, :, :x.size(2)]
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return F.relu(out + residual)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for ET0 forecasting.
    Paper reports 1,617 params — very lightweight.
    Uses dilated causal convolutions.
    """

    def __init__(self, input_size: int, hidden_size: int = 16,
                 num_levels: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        layers = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else hidden_size
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size,
                                        dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        # take last time step
        out = out[:, :, -1]
        out = self.fc(out)
        return out.squeeze(-1)


class LTransformerModel(nn.Module):
    """Lightweight Transformer for ET0 forecasting.
    Paper reports 579 params — tiny.
    Single-head attention, no positional encoding (keep it simple).
    """

    def __init__(self, input_size: int, d_model: int = 16,
                 nhead: int = 1, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=32, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        out = self.transformer(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out.squeeze(-1)


def count_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str, input_size: int, **kwargs) -> nn.Module:
    """Factory function to build models by name."""
    models = {
        'gru': GRUModel,
        'lstm': LSTMModel,
        'rnn': RNNModel,
        'tcn': TCNModel,
        'ltransformer': LTransformerModel,
    }
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name.lower()](input_size=input_size, **kwargs)


# quick test
if __name__ == "__main__":
    input_size = 5  # 5 meteorological features
    seq_len = 12    # 12-hour lookback

    for name in ['gru', 'lstm', 'rnn', 'tcn', 'ltransformer']:
        model = build_model(name, input_size)
        n_params = count_params(model)

        x = torch.randn(4, seq_len, input_size)
        out = model(x)

        print(f"{name:15s} — params: {n_params:>6d}, output: {out.shape}")
        # print(f"  {name} output shape: {out.shape}") # debug shapes

    # print(f"Total params across all models: {sum(count_params(build_model(n, input_size)) for n in ['gru', 'lstm', 'rnn', 'tcn', 'ltransformer'])}") # debug
