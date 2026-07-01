# CHANGELOG.md — Development History

## [1.0.0] - 2026-07-01
- reproduce GRU R²=0.966±0.001 — passes ±5% vs paper 0.9888
- all 5 models trained (GRU, LSTM, RNN, TCN, L-Transformer) × 3 seeds
- ablation: no dropout +0.43%, lookback 6h -0.77%, hidden 32 -0.38%
- extension: hyperparameter sweep 12 configs
- README first draft — tired but done
- fix MAPE calculation — was dividing by near-zero values ugh

## [0.2.0] - 2026-07-01
- ablation sweep complete — 5 variants
- no dropout actually helps on synthetic data (less regularization needed)
- LR 1e-3 diverged on some seeds, rolled back to 5e-4 — TODO clean scheduler later
- TCN surprisingly good — R²=0.967, better than GRU
- L-Transformer paper says <0.90 but we get 0.961 — synthetic data is easier

## [0.1.0] - 2026-07-01
- initial from-scratch train loop — messy but runs
- seed 42 stable across runs
- dataloader off-by-one fixed (was including target in features)
- lr scheduler verbose kwarg removed — pytorch version issue
- smoke test passes on 500 samples, 3 epochs
