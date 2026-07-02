# evaluate.py — evaluation metrics for ET0 forecasting — SR
# R², MAE, RMSE + bootstrap CI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, Optional
import json


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Returns:
        dict with r2, mae, rmse, mse, mape
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # MAPE — avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('nan')

    return {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'mse': float(mse),
        'mape': float(mape),
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_bootstrap: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.

    Returns:
        (mean, lower, upper) of CI
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    scores = np.array(scores)
    mean = scores.mean()
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return float(mean), float(lower), float(upper)


def evaluate_model(model, test_loader, device, scaler_y=None,
                   n_bootstrap: int = 1000) -> Dict:
    """Full evaluation pipeline.

    Args:
        model: trained model
        test_loader: DataLoader for test set
        device: torch device
        scaler_y: inverse scaler for denormalization (optional)
        n_bootstrap: number of bootstrap samples for CI

    Returns:
        dict with all metrics + CI
    """
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)

            y_true_list.append(y_batch.numpy())
            y_pred_list.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # inverse transform if scaler provided
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # bootstrap CI for R²
    r2_mean, r2_lower, r2_upper = bootstrap_ci(
        y_true, y_pred,
        metric_fn=lambda yt, yp: r2_score(yt, yp),
        n_bootstrap=n_bootstrap
    )
    metrics['r2_ci_mean'] = r2_mean
    metrics['r2_ci_lower'] = r2_lower
    metrics['r2_ci_upper'] = r2_upper

    # bootstrap CI for MAE
    mae_mean, mae_lower, mae_upper = bootstrap_ci(
        y_true, y_pred,
        metric_fn=lambda yt, yp: mean_absolute_error(yt, yp),
        n_bootstrap=n_bootstrap
    )
    metrics['mae_ci_mean'] = mae_mean
    metrics['mae_ci_lower'] = mae_lower
    metrics['mae_ci_upper'] = mae_upper

    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
    }


def measure_latency(model, test_loader, device, n_runs: int = 100) -> Dict:
    """Measure inference latency per sample."""
    model.eval()
    latencies = []

    # warmup
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(test_loader):
            if i >= 3:
                break
            X_batch = X_batch.to(device)
            _ = model(X_batch)

    # measure
    import time as time_mod
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            t0 = time_mod.perf_counter()
            _ = model(X_batch)
            t1 = time_mod.perf_counter()
            latencies.append((t1 - t0) * 1000 / X_batch.size(0))  # ms per sample

    latencies = np.array(latencies)
    return {
        'latency_ms_mean': float(np.mean(latencies)),
        'latency_ms_std': float(np.std(latencies)),
        'latency_ms_p50': float(np.percentile(latencies, 50)),
        'latency_ms_p90': float(np.percentile(latencies, 90)),
    }


def print_metrics(metrics: Dict, model_name: str = ""):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results — {model_name}")
    print(f"{'='*60}")
    print(f"R²:   {metrics['metrics']['r2']:.4f} "
          f"[{metrics['metrics']['r2_ci_lower']:.4f}, {metrics['metrics']['r2_ci_upper']:.4f}]")
    print(f"MAE:  {metrics['metrics']['mae']:.6f}")
    print(f"RMSE: {metrics['metrics']['rmse']:.6f}")
    print(f"MSE:  {metrics['metrics']['mse']:.6f}")
    print(f"MAPE: {metrics['metrics']['mape']:.2f}%")
    print(f"{'='*60}")


# CLI entry point
if __name__ == "__main__":
    import argparse
    from src.data import generate_synthetic_data, prepare_data, ET0Dataset
    from src.model import build_model
    from src.utils import set_seed, get_device

    parser = argparse.ArgumentParser(description='Evaluate ET0 forecasting model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=8760)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt['model_name']
    lookback = ckpt.get('lookback', 12)

    # data
    df = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    data = prepare_data(df, lookback=lookback, seed=args.seed)
    test_ds = ET0Dataset(data['X_test'], data['y_test'])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    # model
    model = build_model(model_name, data['n_features'])
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    # evaluate
    results = evaluate_model(model, test_loader, device, scaler_y=data['scaler_y'])
    print_metrics(results, model_name)

    # latency
    latency = measure_latency(model, test_loader, device)
    print(f"\nLatency: {latency['latency_ms_mean']:.3f} ± {latency['latency_ms_std']:.3f} ms/sample")
