# run_experiments.py — run all models × seeds — SR
# from-scratch reproduction of Luo et al. 2025

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model
from src.evaluate import evaluate_model, measure_latency, print_metrics
from src.model import count_params, build_model
from src.utils import set_seed, get_device, save_results_csv
import torch
import numpy as np
import json
import time

MODELS = ['gru', 'lstm', 'rnn', 'tcn', 'ltransformer']
SEEDS = [42, 123, 2026]
LOOKBACK = 12
EPOCHS = 150
BATCH_SIZE = 64
LR = 5e-4
N_SAMPLES = 8760  # 1 year hourly


def run_single_experiment(model_name, seed, lookback=LOOKBACK,
                          epochs=EPOCHS, n_samples=N_SAMPLES):
    """Run single model×seed experiment."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} | seed={seed}")
    print(f"{'='*60}")

    start_time = time.time()

    # train
    model, history, data, test_loader = train_model(
        model_name=model_name,
        lookback=lookback,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=seed,
        save_dir=f'runs/{model_name}_seed{seed}',
        n_samples=n_samples,
        patience=25,
    )

    train_time = time.time() - start_time

    # evaluate
    device = get_device()
    # reload best model
    ckpt = torch.load(f'runs/{model_name}_seed{seed}/best_{model_name}_seed{seed}.pt',
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    results = evaluate_model(model, test_loader, device, scaler_y=data['scaler_y'])
    latency = measure_latency(model, test_loader, device)

    n_params = count_params(model)

    print_metrics(results, f"{model_name} (seed={seed})")
    print(f"Latency: {latency['latency_ms_mean']:.3f} ms/sample")
    print(f"Train time: {train_time:.1f}s")

    return {
        'model': model_name,
        'seed': seed,
        'lookback': lookback,
        'epochs_ran': len(history['train_loss']),
        'train_time_s': round(train_time, 1),
        'train_loss_final': round(history['train_loss'][-1], 6),
        'val_loss_final': round(history['val_loss'][-1], 6),
        'r2': round(results['metrics']['r2'], 4),
        'r2_ci_lower': round(results['metrics']['r2_ci_lower'], 4),
        'r2_ci_upper': round(results['metrics']['r2_ci_upper'], 4),
        'mae': round(results['metrics']['mae'], 6),
        'rmse': round(results['metrics']['rmse'], 6),
        'mse': round(results['metrics']['mse'], 6),
        'mape': round(results['metrics']['mape'], 2),
        'latency_ms_mean': round(latency['latency_ms_mean'], 3),
        'latency_ms_std': round(latency['latency_ms_std'], 3),
        'params': n_params,
        'ram_peak_mb': None,  # would need psutil monitoring
    }


def run_all_experiments():
    """Run full experiment grid: all models × all seeds."""
    all_results = []

    for model_name in MODELS:
        for seed in SEEDS:
            result = run_single_experiment(model_name, seed)
            all_results.append(result)

    # save results
    os.makedirs('experiments', exist_ok=True)
    save_results_csv(all_results, 'experiments/results_original.csv')

    # summary table
    print(f"\n{'='*80}")
    print("SUMMARY — All Models × All Seeds")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Seed':<6} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'Latency':<10} {'Params':<8}")
    print(f"{'-'*15} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for r in all_results:
        print(f"{r['model']:<15} {r['seed']:<6} {r['r2']:<8.4f} {r['mae']:<10.6f} "
              f"{r['rmse']:<10.6f} {r['latency_ms_mean']:<10.3f} {r['params']:<8}")

    # aggregate by model
    print(f"\n{'='*80}")
    print("AGGREGATED — mean ± std across seeds")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'R² mean±std':<20} {'MAE mean±std':<25} {'RMSE mean±std':<25}")
    print(f"{'-'*15} {'-'*20} {'-'*25} {'-'*25}")

    for model_name in MODELS:
        model_results = [r for r in all_results if r['model'] == model_name]
        r2_vals = [r['r2'] for r in model_results]
        mae_vals = [r['mae'] for r in model_results]
        rmse_vals = [r['rmse'] for r in model_results]

        print(f"{model_name:<15} {np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}      "
              f"{np.mean(mae_vals):.6f}±{np.std(mae_vals):.6f}  "
              f"{np.mean(rmse_vals):.6f}±{np.std(rmse_vals):.6f}")

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
