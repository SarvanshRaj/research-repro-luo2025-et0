# data.py — data loading + sliding window for ET0 forecasting
# reimplemented from Luo et al. 2025 — SR

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import os

# seed control
np.random.seed(42)


def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load raw meteorological data. Returns DataFrame with timestamp index."""
    # placeholder — will be filled with actual data loading
    # paper uses Sichuan meteorological + soil sensor data
    # features: temperature, humidity, wind speed, solar radiation, soil moisture, etc.
    raise NotImplementedError("Download data first — see data/README.md")


def generate_synthetic_data(n_samples: int = 8760, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ET0-like data for development/testing.

    Creates hourly meteorological data for one year (8760 hours).
    Patterns mimic real ET0: diurnal cycle + seasonal + noise.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples)

    # diurnal cycle (24h period)
    diurnal = np.sin(2 * np.pi * t / 24) * 0.3

    # seasonal cycle (365 days)
    seasonal = np.sin(2 * np.pi * t / (365 * 24)) * 0.4 + 0.5

    # temperature (°C) — correlates with ET0
    temp = 15 + 10 * seasonal + 5 * diurnal + rng.normal(0, 2, n_samples)

    # humidity (%) — inverse correlation with temp
    humidity = 80 - 20 * seasonal - 10 * diurnal + rng.normal(0, 5, n_samples)
    humidity = np.clip(humidity, 10, 100)

    # wind speed (m/s)
    wind = 2 + 1.5 * rng.exponential(1, n_samples) + 0.5 * diurnal
    wind = np.clip(wind, 0, 15)

    # solar radiation (W/m²) — strong diurnal
    solar = np.maximum(0, 500 * seasonal + 300 * diurnal + rng.normal(0, 50, n_samples))

    # soil moisture (%)
    soil_moisture = 30 + 10 * seasonal - 5 * diurnal + rng.normal(0, 3, n_samples)
    soil_moisture = np.clip(soil_moisture, 5, 60)

    # ET0 target — Penman-Monteith simplified
    # paper uses actual ET0 from weather station
    et0 = (0.0023 * (temp + 17.8) * np.sqrt(np.maximum(0, temp + 5)) *
           (0.5 + 0.537 * wind) * solar / 1000)
    et0 = np.maximum(0, et0 + rng.normal(0, 0.02, n_samples))

    df = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity,
        'wind_speed': wind,
        'solar_radiation': solar,
        'soil_moisture': soil_moisture,
        'et0': et0
    })

    return df


def create_sequences(data: np.ndarray, lookback: int = 12,
                     horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for time series forecasting.

    Args:
        data: array of shape (n_samples, n_features)
        lookback: number of past time steps as input
        horizon: number of future steps to predict (default 1)

    Returns:
        X: (n_sequences, lookback, n_features)
        y: (n_sequences,)
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, -1])  # last col = ET0

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    if horizon == 1:
        y = y.squeeze(-1)

    return X, y


def prepare_data(df: pd.DataFrame, lookback: int = 12,
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 seed: int = 42) -> dict:
    """Prepare train/val/test splits with scaling.

    Paper uses 70/15/15 split approximately.
    """
    rng = np.random.RandomState(seed)

    # features: all columns except et0
    feature_cols = [c for c in df.columns if c != 'et0']
    target_col = 'et0'

    # scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df[feature_cols].values)
    y_scaled = scaler_y.fit_transform(df[[target_col]].values)

    # create sequences from features only, target separate
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - lookback):
        X_seq.append(X_scaled[i:i + lookback])
        y_seq.append(y_scaled[i + lookback, 0])  # next step ET0

    X = np.array(X_seq, dtype=np.float32)
    y = np.array(y_seq, dtype=np.float32)

    # split (chronological, not random — time series)
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'lookback': lookback,
        'n_features': X_train.shape[2]
    }


class ET0Dataset:
    """PyTorch-style dataset for ET0 sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# quick test
if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=1000)
    print(f"Generated data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"ET0 range: {df['et0'].min():.4f} to {df['et0'].max():.4f}")

    data = prepare_data(df, lookback=12)
    print(f"Train X: {data['X_train'].shape}, y: {data['y_train'].shape}")
    print(f"Val X: {data['X_val'].shape}, y: {data['y_val'].shape}")
    print(f"Test X: {data['X_test'].shape}, y: {data['y_test'].shape}")
    print(f"Features: {data['n_features']}")
