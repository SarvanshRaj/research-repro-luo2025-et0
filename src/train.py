# train.py — training loop for ET0 forecasting — SR
# from-scratch reproduction of Luo et al. 2025

import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from typing import Optional, Dict
from datetime import datetime

from src.data import generate_synthetic_data, prepare_data, ET0Dataset
from src.model import build_model, count_params
from src.utils import set_seed, get_device, AverageMeter


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Single training epoch."""
    model.train()
    loss_meter = AverageMeter()

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        # gradient clipping — helps stabilize RNN training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), X_batch.size(0))

    return loss_meter.avg


def validate(model, val_loader, criterion, device):
    """Validation loop."""
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss_meter.update(loss.item(), X_batch.size(0))

    return loss_meter.avg


def train_model(model_name: str = 'gru', lookback: int = 12,
                epochs: int = 100, batch_size: int = 32,
                lr: float = 1e-3, seed: int = 42,
                save_dir: str = 'runs', n_samples: int = 8760,
                patience: int = 15, **model_kwargs):
    """Full training pipeline.

    Args:
        model_name: one of 'gru', 'lstm', 'rnn', 'tcn', 'ltransformer'
        lookback: input sequence length (paper tests 6, 12, 24)
        epochs: max training epochs
        batch_size: batch size
        lr: learning rate
        seed: random seed for reproducibility
        save_dir: directory to save checkpoints
        n_samples: number of synthetic data samples (default 8760 = 1 year hourly)
        patience: early stopping patience
        **model_kwargs: additional model hyperparameters
    """
    # seed everything
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {model_name}, Lookback: {lookback}, Seed: {seed}")

    # data
    print("Generating data...")
    df = generate_synthetic_data(n_samples=n_samples, seed=seed)
    data = prepare_data(df, lookback=lookback, seed=seed)

    train_ds = ET0Dataset(data['X_train'], data['y_train'])
    val_ds = ET0Dataset(data['X_val'], data['y_val'])
    test_ds = ET0Dataset(data['X_test'], data['y_test'])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    # model
    model = build_model(model_name, data['n_features'], **model_kwargs)
    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")

    model = model.to(device)

    # loss + optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # LR scheduler — paper might use step decay
    # TODO: clean LR schedule later — paper doesn't specify clearly
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # training loop
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        scheduler.step(val_loss)

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_name': model_name,
                'n_params': n_params,
                'seed': seed,
                'lookback': lookback,
            }, os.path.join(save_dir, f'best_{model_name}_seed{seed}.pt'))

        # logging
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | Best: {best_val_loss:.6f} @ {best_epoch} | "
                  f"Time: {elapsed:.1f}s")

        # early stopping
        if epoch - best_epoch > patience:
            print(f"Early stopping at epoch {epoch} (best @ {best_epoch})")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best val loss: {best_val_loss:.6f} @ epoch {best_epoch}")

    # save history
    with open(os.path.join(save_dir, f'history_{model_name}_seed{seed}.json'), 'w') as f:
        json.dump(history, f)

    return model, history, data, test_loader


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train ET0 forecasting model')
    parser.add_argument('--model', type=str, default='gru',
                        choices=['gru', 'lstm', 'rnn', 'tcn', 'ltransformer'])
    parser.add_argument('--lookback', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='runs')
    parser.add_argument('--n_samples', type=int, default=8760)

    args = parser.parse_args()

    model, history, data, test_loader = train_model(
        model_name=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        n_samples=args.n_samples,
    )
