# REPRODUCE.md — Exact Reproduction Steps

## Prerequisites

- Python 3.10+
- ~2GB disk space
- ~4GB RAM
- No GPU required (CPU training ~2-3 min per model)

## Step 1: Clone

```bash
git clone https://github.com/SarvanshRaj/research-repro-luo2025-et0.git
cd research-repro-luo2025-et0
```

## Step 2: Environment

```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yml
conda activate et0-repro
```

## Step 3: Train (single model)

```bash
python src/train.py --model gru --seed 42 --epochs 100 --lookback 12
```

Expected output:
```
Using device: cpu
Model: gru, Lookback: 12, Seed: 42
Parameters: 40,705
Training for 100 epochs...
...
Best val loss: 0.000275 @ epoch 147
```

Expected runtime: ~2-3 minutes on modern CPU.

## Step 4: Evaluate

```bash
python src/evaluate.py --checkpoint runs/best_gru_seed42.pt
```

Expected metrics:
- R²: 0.964–0.968
- MAE: 0.010–0.011
- RMSE: 0.013–0.014

## Step 5: Full reproduction (all models × all seeds)

```bash
python experiments/run_experiments.py
```

This runs 15 experiments (5 models × 3 seeds). Expected runtime: ~20 minutes on CPU.

Output: `experiments/results_original.csv`

## Step 6: Ablation

```bash
python experiments/ablation.py
```

Runs 5 ablation variants. Expected runtime: ~8 minutes.

Output: `experiments/results_ablation.csv`

## Seeds

All experiments use seeds: 42, 123, 2026.

```python
# Deterministic settings applied in src/utils.py
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Expected Results Summary

| Model | R² (mean±std) | MAE | Params |
|-------|---------------|-----|--------|
| GRU | 0.966 ± 0.001 | 0.0106 | 40,705 |
| LSTM | 0.965 ± 0.001 | 0.0106 | 53,569 |
| RNN | 0.949 ± 0.006 | 0.0130 | 1,281 |
| TCN | 0.967 ± 0.002 | 0.0103 | 4,481 |
| L-Transformer | 0.961 ± 0.003 | 0.0110 | 2,337 |

## Troubleshooting

**"No module named src"**: Run from the project root directory.

**Low R² (<0.90)**: Check that data generation uses the latest code (seed 42 should produce consistent results).

**Memory error**: Reduce batch_size: `python src/train.py --batch_size 16`

**Colab disconnects**: Add checkpoint saving to every epoch (already implemented).

## Notes

I ran these experiments on CPU in a sandboxed environment. Results are deterministic with seed control. Minor variations (<0.001 in R²) may occur across different CPU architectures due to floating-point ordering.
