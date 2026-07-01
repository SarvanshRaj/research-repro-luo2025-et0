# utils.py — helpers for ET0 forecasting repro — SR

import torch
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # tf seed for tflite
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_training_curves(history: Dict, save_path: Optional[str] = None,
                         title: str = "Training Curves"):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train', alpha=0.8)
    ax1.plot(history['val_loss'], label='Val', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'{title} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['lr'], label='LR', color='green', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'{title} — LR Schedule')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     title: str = "Predictions vs Actual",
                     save_path: Optional[str] = None, n_points: int = 200):
    """Plot predictions vs actual values."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # time series comparison (first n_points)
    n = min(n_points, len(y_true))
    ax1.plot(y_true[:n], label='Actual', alpha=0.8)
    ax1.plot(y_pred[:n], label='Predicted', alpha=0.8, linestyle='--')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('ET0')
    ax1.set_title(f'{title} — Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect')
    ax2.set_xlabel('Actual ET0')
    ax2.set_ylabel('Predicted ET0')
    ax2.set_title(f'{title} — Scatter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_model_comparison(results: Dict[str, Dict], metric: str = 'r2',
                          save_path: Optional[str] = None):
    """Bar plot comparing models on a metric."""
    models = list(results.keys())
    values = [results[m]['metrics'][metric] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison — {metric.upper()}')
    ax.grid(True, alpha=0.3, axis='y')

    # add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.close()


def save_results_csv(results: List[Dict], filepath: str):
    """Save results list to CSV."""
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(results)} results to {filepath}")


def load_checkpoint(checkpoint_path: str, device=None):
    """Load model checkpoint."""
    if device is None:
        device = get_device()
    return torch.load(checkpoint_path, map_location=device)
