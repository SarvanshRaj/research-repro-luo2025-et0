# GATE-B — 3-Finalist Breakdown

## Finalist #1: P-B — Luo et al., 2025, Hydrology Research
**"Benchmarking deep learning models for ET0 forecasting on edge devices"**

### 9-Criteria Breakdown /100 + Bonus

| # | Criteria | Score | Evidence |
|---|----------|-------|----------|
| 1 | Top-Uni + recruiter admissions fit | 19/20 | Direct bridge: Biogas predictive AI (load forecasting) → ET0 (energy/water forecasting). IoT sensor pipeline extension. Maps NUS/NTU CE, TUM EE/Informatics, MEXT Natural Sciences, FH Upper Austria AI Solutions. |
| 2 | Resume / portfolio continuity | 14/15 | Extends BOTH portfolio anchors: IoT DHT11→Pi sensor pipeline (data collection) + Biogas predictive AI (supervised learning forecasting). |
| 3 | Reproducibility & openness | 14/15 | Open data (Sichuan meteorological), open preprocessing/deployment scripts, TFLite quantized models released. Hydrology Research OA (CC-BY). |
| 4 | From-scratch feasibility | 8/10 | GRU/LSTM/TCN/RNN well-documented architectures. ~500-600 LOC core. PyTorch or TF/Keras. ≤6h Colab T4. |
| 5 | Free-forever compute | 9/10 | Colab Free T4 sufficient. Training ~1-2h. Inference on Pi4B (simulateable). RAM <4GB. |
| 6 | Measurable benchmark quality | 9/10 | R²=0.9888, MAE=0.0108, RMSE reported. Latency 1.33ms/sample. Memory 638MB. Model size 216.96KB TFLite. Dataset >1k samples. |
| 7 | Extension potential | 9/10 | New climate dataset transfer. Hyperparam sweep (window length 6/12/24h). TFLite INT8 quantization. Edge deploy on Pi. |
| 8 | Documentation | 4/5 | Hydrology Research — peer-reviewed. Good methodological detail. |
| 9 | License / ethics | 4/5 | CC-BY. No PII. Agricultural/environmental data. |
| **Bonus** | Universal job-market relevance | +4 | Precision agriculture, campus energy labs, smallholder irrigation IoT, green AI forecasting. |
| **TOTAL** | | **93/100** | |

### Risk Register — Top 3

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Sichuan dataset not directly downloadable (paper supplementary only) | Medium | High | Contact authors OR use alternative open ET0 dataset (FAO/FAOSTAT, NASA POWER ag data). Verify downloadability before build. |
| R2 | GRU architecture details not fully specified (hidden units, layers, dropout) | Low | Medium | Infer from param count (25,313) + standard GRU configs. Use paper's Table 1 if available. |
| R3 | Pi4B hardware not available for full edge benchmark | Low | Low | Simulate latency on Colab CPU (proxy). Document Pi4B target specs. Energy estimate from param count + literature. |

### From-Scratch Plan — Modules to Author ≥85%

| Module | LOC Est | Author % | Notes |
|--------|---------|----------|-------|
| src/data.py — data loader + windowing | 80 | 100% | Custom sliding window from numpy arrays |
| src/model.py — GRU/LSTM/TCN/RNN | 120 | 95% | PyTorch nn.GRU/LSTM. TCN from scratch (dilated conv). 5%: standard conv layers. |
| src/train.py — training loop | 100 | 100% | From scratch: seed, checkpointing, LR schedule |
| src/evaluate.py — metrics | 60 | 100% | R², MAE, RMSE, bootstrap CI |
| src/export_tflite.py — TFLite conversion | 40 | 90% | TFLiteConverter API usage. 10%: API boilerplate. |
| src/utils.py — helpers | 50 | 100% | Seed, plotting, logging |
| notebooks/research_colab.ipynb | N/A | 90% | End-to-end. 10%: Colab boilerplate. |
| **Total core** | **~450** | **~97%** | Attributed: TFLiteConverter API, PyTorch nn.Module base |

**Attributed LOC target:** ≤15% (currently estimated ~3%)

### Compute Budget

| Resource | Estimated | Tool | Cost |
|----------|-----------|------|------|
| GPU Training | ~2h T4 | Colab Free | $0 |
| Data preprocessing | ~0.5h CPU | Colab Free | $0 |
| Evaluation + plots | ~0.5h CPU | Colab Free | $0 |
| Ablation sweep | ~2h T4 | Kaggle P100 (backup) | $0 |
| Extension experiment | ~1.5h T4 | Colab Free | $0 |
| **Total** | **~6.5h** | | **$0** |

**RAM:** <4GB peak. **Disk:** <500MB (data + models).

### Measurable Benchmark SLO

```
Primary: R² ≥ 0.984 ± 0.005 (n_seeds=3)
Secondary: MAE ≤ 0.012 ± 0.002
Secondary: RMSE ≤ 0.020 ± 0.003
Latency (TFLite CPU): ≤ 2.0 ms/sample
Params: ≤ 30,000
Model size (TFLite): ≤ 250 KB
```

---

## Finalist #2: P-C — Power-Aware Edge Inference, 2026, TechRxiv
**"Power-Aware Image Classification on a Low-Cost Edge SBC"**

### 9-Criteria Breakdown /100 + Bonus

| # | Criteria | Score | Evidence |
|---|----------|-------|----------|
| 1 | Admissions fit | 17/20 | Robotics perception. Green AI energy efficiency. Less direct Biogas/sustainable link than P-B. |
| 2 | Resume continuity | 11/15 | IoT Env Monitoring extends (sensor → edge compute). Weak on Biogas. |
| 3 | Reproducibility | 14/15 | Fully open CSV logs, scripts, figures. Bit-identical reproduction without hardware. |
| 4 | From-scratch feasibility | 7/10 | MobileNetV3-Small + ResNet-18 — larger models (~2.5M + ~11M params). Needs torchvision or from-scratch. ~800 LOC. |
| 5 | Free-forever compute | 8/10 | Training ~3h T4. Energy measurement needs INA219 (hardware ~$5). Can simulate. |
| 6 | Benchmark quality | 9/10 | Accuracy, latency, energy mJ/inf, FPS/W. Multiple datasets. Measured with INA219. |
| 7 | Extension potential | 8/10 | New model architectures, INT8 quantization, different SBCs. |
| 8 | Documentation | 3/5 | TechRxiv preprint — not peer-reviewed. Good technical detail. |
| 9 | License | 4/5 | CC-BY 4.0 TechRxiv. |
| **Bonus** | | +5 | Edge robotics, autonomous systems, green AI — highest universal job-market relevance. |
| **TOTAL** | | **86/100** | |

### Risk Register — Top 3

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | MobileNetV3-Small from scratch complex (~2.5M params) | Medium | High | Use torchvision.models + reproduce benchmark harness. Architecture is public. |
| R2 | INA219 hardware needed for energy measurement | Low | Medium | Simulate energy from latency × literature power values. Document assumptions. |
| R3 | CIFAR-100/SVHN may need GPU for full training | Low | Low | Colab Free T4 handles both easily. |

### Compute Budget
- Training: ~3h T4 | Colab Free | $0
- Total: ~5h | $0

### Benchmark SLO
```
Primary: MobileNetV3-Small accuracy ≥ 70% ±2% on CIFAR-100
Secondary: Latency ≤ 3.0 ms (Colab CPU proxy)
Secondary: Energy ≤ 8 mJ/inf (estimated from latency)
Params: MobileNetV3-Small ~2.5M
```

---

## Finalist #3: P-D — TinyML Acoustic Anomaly Detection, 2026, arXiv
**"TinyML for Acoustic Anomaly Detection in IoT Sensor Networks"**

### 9-Criteria Breakdown /100 + Bonus

| # | Criteria | Score | Evidence |
|---|----------|-------|----------|
| 1 | Admissions fit | 16/20 | IoT sensor networks. Acoustic monitoring. Less direct energy/sustainable link. |
| 2 | Resume continuity | 10/15 | IoT Env Monitoring extends (sensor → acoustic). Weak on Biogas. |
| 3 | Reproducibility | 12/15 | UrbanSound8K well-known open dataset. arXiv preprint — code may be available. |
| 4 | From-scratch feasibility | 9/10 | Small model (61k params). MFCC + compact NN. ~300-400 LOC. Very fast training. |
| 5 | Free-forever compute | 10/10 | ~1h Colab Free. RAM <2GB. Easiest to reproduce. |
| 6 | Benchmark quality | 8/10 | 91% accuracy, F1=0.91. UrbanSound8K 8,732 samples. |
| 7 | Extension potential | 7/10 | Multi-class classification. Different audio datasets. TFLite Micro deploy. |
| 8 | Documentation | 3/5 | arXiv preprint — ICECCME conference. Limited detail. |
| 9 | License | 3/5 | arXiv preprint — license unclear. UrbanSound8K CC-BY. |
| **Bonus** | | +3 | Smart city, campus safety, environmental monitoring. |
| **TOTAL** | | **81/100** | |

### Risk Register — Top 3

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Paper details sparse (arXiv preprint) | Medium | Medium | Use standard MFCC + compact NN pipeline. UrbanSound8K well-documented elsewhere. |
| R2 | Binary classification grouping not fully specified | Low | Low | Group urban sounds into normal/anomalous per paper's description. |
| R3 | No peer review yet | Low | Low | Reproduce and validate independently. |

### Compute Budget
- Training: ~1h T4 | Colab Free | $0
- Total: ~2h | $0

### Benchmark SLO
```
Primary: Test accuracy ≥ 86% ±5% (n_seeds=3)
Secondary: F1 ≥ 0.86 ±0.05
Params: ~62,000
Model size (INT8): ≤ 70 KB
```

---

## RECOMMENDATION

**P-B (Luo et al., 2025 — Edge ET0 Forecasting) is the strongest choice** for three reasons:

1. **Universal technical impact:** Edge AI for precision agriculture/irrigation is globally relevant — from campus labs to smallholder farms. Demonstrates complete ML pipeline: data preprocessing → model training → TFLite quantization → Raspberry Pi deployment.

2. **Resume continuity:** Directly extends BOTH portfolio anchors — IoT Environmental Monitoring (sensor data pipeline) and Biogas Optimization (supervised learning forecasting). Creates a coherent narrative: sensors → data → AI forecasting → edge deployment.

3. **Measurable benchmark feasibility:** Primary metric R²=0.9888 with clear ±5% threshold. Five model architectures (GRU/LSTM/TCN/RNN/L-Transformer) enable ablation without external dependencies. ~450 LOC core, ~6.5h total compute, $0 — feasible from scratch within constraints.

---

=== GATE-B AWAITING APPROVAL: 3-FINALIST + RECOMMENDATION ===

## Approval Status

To proceed to build: reply "APPROVE: P-B" (or alternative finalist ID).
