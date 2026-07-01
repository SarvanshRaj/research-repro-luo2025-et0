# ablation.py — ablation experiments — SR
# variations: no dropout, different lookback, different hidden size

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model
from src.evaluate import evaluate_model, print_metrics
from src.utils import get_device, save_results_csv
import torch
import numpy as np
import time

SEED = 42
N_SAMPLES = 8760


def run_ablation():
    """Run ablation variants."""
    ablation_configs = [
        # baseline (already have)
        {'name': 'baseline_gru', 'model': 'gru', 'lookback': 12,
         'hidden_size': 64, 'dropout': 0.1, 'epochs': 150},

        # no dropout
        {'name': 'gru_no_dropout', 'model': 'gru', 'lookback': 12,
         'hidden_size': 64, 'dropout': 0.0, 'epochs': 150},

        # shorter lookback (6h)
        {'name': 'gru_lookback6', 'model': 'gru', 'lookback': 6,
         'hidden_size': 64, 'dropout': 0.1, 'epochs': 150},

        # smaller hidden (32)
        {'name': 'gru_hidden32', 'model': 'gru', 'lookback': 12,
         'hidden_size': 32, 'dropout': 0.1, 'epochs': 150},

        # LSTM instead of GRU
        {'name': 'baseline_lstm', 'model': 'lstm', 'lookback': 12,
         'hidden_size': 64, 'dropout': 0.1, 'epochs': 150},
    ]

    results = []
    for cfg in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Ablation: {cfg['name']}")
        print(f"{'='*60}")

        start = time.time()
        model, history, data, test_loader = train_model(
            model_name=cfg['model'],
            lookback=cfg['lookback'],
            epochs=cfg['epochs'],
            batch_size=64,
            lr=5e-4,
            seed=SEED,
            save_dir=f'runs/ablation_{cfg["name"]}',
            n_samples=N_SAMPLES,
            patience=25,
            hidden_size=cfg.get('hidden_size', 64),
            dropout=cfg.get('dropout', 0.1),
        )
        train_time = time.time() - start

        device = get_device()
        ckpt = torch.load(
            f'runs/ablation_{cfg["name"]}/best_{cfg["model"]}_seed{SEED}.pt',
            map_location=device
        )
        model.load_state_dict(ckpt['model_state_dict'])
        eval_results = evaluate_model(model, test_loader, device,
                                      scaler_y=data['scaler_y'])

        from src.model import count_params
        r = {
            'variant': cfg['name'],
            'model': cfg['model'],
            'lookback': cfg['lookback'],
            'hidden_size': cfg.get('hidden_size', 64),
            'dropout': cfg.get('dropout', 0.1),
            'r2': round(eval_results['metrics']['r2'], 4),
            'mae': round(eval_results['metrics']['mae'], 6),
            'rmse': round(eval_results['metrics']['rmse'], 6),
            'params': count_params(model),
            'train_time_s': round(train_time, 1),
        }
        results.append(r)
        print(f"  R²: {r['r2']:.4f}, MAE: {r['mae']:.6f}, RMSE: {r['rmse']:.6f}")

    # save
    save_results_csv(results, 'experiments/results_ablation.csv')

    # print table
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Variant':<25} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'Params':<8} {'Time':<8}")
    print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    baseline_r2 = results[0]['r2']
    for r in results:
        delta = r['r2'] - baseline_r2
        print(f"{r['variant']:<25} {r['r2']:<8.4f} {r['mae']:<10.6f} "
              f"{r['rmse']:<10.6f} {r['params']:<8} {r['train_time_s']:<8.1f}")

    # delta from baseline
    print(f"\nDelta from baseline (R²={baseline_r2:.4f}):")
    for r in results[1:]:
        delta = r['r2'] - baseline_r2
        print(f"  {r['variant']:<25}: {delta:+.4f} ({delta/baseline_r2*100:+.2f}%)")

    return results


if __name__ == "__main__":
    run_ablation()
