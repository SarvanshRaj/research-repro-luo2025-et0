# Paper Summary — Luo et al., 2025

## Title
Benchmarking deep learning models for ET0 forecasting on edge devices

## Authors
Kai Luo, Cheng Siong Lim, Mohamad Hafis Izran Bin Ishak, Mohd Saiful Azimi Mahmud, Ni Ba

## Year / Venue
2025 / Hydrology Research (IWA Publishing, Vol 56, Issue 12, pp 1269–1298)

## DOI
https://doi.org/10.2166/nh.2025.130

## License
Open Access (CC-BY) — Hydrology Research

## Problem (≤80 words)
Edge AI for precision irrigation needs lightweight forecasting models that run on Raspberry Pi-class hardware with limited memory and power. This paper benchmarks five deep learning architectures (GRU, LSTM, RNN, TCN, L-Transformer) for ET0 (reference evapotranspiration) prediction using multivariate meteorological + soil sensor data, deployed via TFLite on Raspberry Pi 4B. The goal is to identify which architecture provides the best accuracy-efficiency tradeoff for real-world bandwidth-limited agricultural IoT deployments.

## Dataset(s)
- **Name:** Sichuan, China meteorological + soil sensor data (hourly)
- **n_samples:** ~8,760 (1 year, hourly readings)
- **Features:** 5 — temperature, humidity, wind speed, solar radiation, soil moisture
- **Target:** ET0 (reference evapotranspiration, computed via FAO-56 Penman-Monteith)
- **Size:** <10 MB
- **License:** Available with paper (supplementary material)
- **Preprocessing:** IQR outlier detection, median filtering, Savitzky-Golay smoothing, MinMax scaling

## Model Architecture Table

| Model | Params | TFLite Size (KB) | Real-Time Memory (MB) | Peak Memory (MB) |
|-------|--------|-----------------|----------------------|-----------------|
| GRU | 25,313 | 216.96 | 638.8 | 673.5 |
| LSTM | 30,625 | 187.27 | 522.68 | 530.62 |
| RNN | 2,049 | 32.28 | 487.32 | 489.50 |
| TCN | 1,617 | 9.34 | 466.42 | 466.71 |
| L-Transformer | 579 | 11.79 | 487.51 | 487.95 |

## Evaluation Metrics
- **Primary:** R² (coefficient of determination) — paper reports GRU: 0.9888
- **Secondary 1:** MAE (mean absolute error) — GRU: 0.0108
- **Secondary 2:** RMSE (root mean squared error) — reported
- **Deployment:** Inference latency ms/sample, memory MB, TFLite model size KB

## Algorithm Pseudocode
1. Load multivariate meteorological + soil sensor data
2. Compute ET0 via FAO-56 Penman-Monteith equation
3. Preprocess: outlier detection (IQR), smoothing (Savitzky-Golay), normalize (MinMax)
4. Create sliding window sequences (lookback: 6/12/24h, horizon: 1h)
5. Split: 70% train / 15% val / 15% test (chronological)
6. Train model (GRU/LSTM/RNN/TCN/L-Transformer) with MSE loss + Adam optimizer
7. Evaluate: R², MAE, RMSE on test set
8. Export to TFLite (post-training quantization)
9. Deploy on Raspberry Pi 4B, measure latency + memory

## Implementation Checklist
- [ ] Data loader with sliding window (data.py)
- [ ] MinMax scaling (data.py)
- [ ] GRU model (model.py)
- [ ] LSTM model (model.py)
- [ ] RNN model (model.py)
- [ ] TCN with dilated causal convolutions (model.py)
- [ ] L-Transformer model (model.py)
- [ ] MSE loss + Adam optimizer (train.py)
- [ ] Training loop with early stopping (train.py)
- [ ] Evaluation: R², MAE, RMSE (evaluate.py)
- [ ] Bootstrap CI for metrics (evaluate.py)
- [ ] TFLite export (export_tflite.py)
- [ ] Latency measurement (evaluate.py)
- [ ] Visualization: training curves, predictions, model comparison (utils.py)

## Hardware/Software Requirements
- Python 3.10+
- PyTorch ≥2.0 (training)
- TensorFlow ≥2.15 (TFLite export)
- scikit-learn ≥1.3
- Colab Free T4 (training) — RAM <4GB
- Raspberry Pi 4B (deployment target)

## Reproducibility Risks
1. **Data availability:** Sichuan dataset may be paper-supplementary only. Mitigation: generate realistic synthetic data for development, document data gap.
2. **Hyperparameter specifics:** Paper may not specify all hyperparams (hidden size, layers, dropout). Mitigation: infer from param count, standard configs.
3. **TFLite quantization:** Paper uses post-training dynamic range quantization. Straightforward to reproduce.

## Measurable Benchmark SLO
```
Primary: R² ≥ 0.984 ± 0.005 (n_seeds=3)
Secondary: MAE ≤ 0.012 ± 0.002
Secondary: RMSE ≤ 0.020 ± 0.003
Latency (Colab CPU proxy): ≤ 2.0 ms/sample
Params (GRU): ≤ 30,000
Model size (TFLite): ≤ 250 KB
```

## Universal Use-Case Note
Smallholder irrigation sensor nodes, campus energy labs, agricultural IoT startups, student edge-AI teaching, water management in developing regions.
