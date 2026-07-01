# Task Decomposition — WBS — From Scratch

## T-01: Data Loader + Synthetic Data
- **Status:** [x] DONE
- **File:** src/data.py
- **Est hrs:** 1.0
- **Dependencies:** none
- **Acceptance:** generates synthetic data, sliding window works, shapes correct
- **Author:** from-scratch
- **LOC est:** 80

## T-02: Preprocessing + Feature Scaling
- **Status:** [x] DONE
- **File:** src/data.py
- **Est hrs:** 0.5
- **Dependencies:** T-01
- **Acceptance:** MinMaxScaler applied, train/val/test split chronological
- **Author:** from-scratch
- **LOC est:** 40

## T-03: Model Implementations
- **Status:** [x] DONE
- **File:** src/model.py
- **Est hrs:** 2.0
- **Dependencies:** none
- **Acceptance:** GRU, LSTM, RNN, TCN, L-Transformer all forward-pass correctly, param counts reasonable
- **Author:** from-scratch (95% — standard nn.Module boilerplate 5%)
- **LOC est:** 120

## T-04: Loss + Optimizer
- **Status:** [x] DONE
- **File:** src/train.py
- **Est hrs:** 0.5
- **Dependencies:** T-03
- **Acceptance:** MSE loss, Adam optimizer, gradient clipping
- **Author:** from-scratch
- **LOC est:** 20

## T-05: Training Loop + Checkpointing
- **Status:** [x] DONE
- **File:** src/train.py
- **Est hrs:** 1.5
- **Dependencies:** T-01, T-03, T-04
- **Acceptance:** trains to convergence, saves best checkpoint, early stopping, seed control
- **Author:** from-scratch
- **LOC est:** 100

## T-06: Evaluation — Metrics + Bootstrap CI
- **Status:** [x] DONE
- **File:** src/evaluate.py
- **Est hrs:** 1.5
- **Dependencies:** T-05
- **Acceptance:** R², MAE, RMSE computed correctly, bootstrap CI with n=1000
- **Author:** from-scratch (95% — sklearn API 5%)
- **LOC est:** 60

## T-07: Visualization
- **Status:** [x] DONE
- **File:** src/utils.py
- **Est hrs:** 1.0
- **Dependencies:** T-06
- **Acceptance:** training curves, predictions vs actual, model comparison bar chart
- **Author:** from-scratch
- **LOC est:** 50

## T-08: TFLite Export
- **Status:** [~] WIP
- **File:** src/export_tflite.py
- **Est hrs:** 1.0
- **Dependencies:** T-05
- **Acceptance:** ONNX export works, TFLite export (may need fallback), latency measurement
- **Author:** from-scratch (90% — TFLite API boilerplate 10%)
- **LOC est:** 40

## T-09: Ablation Harness
- **Status:** [ ] TODO
- **File:** src/train.py (sweep mode)
- **Est hrs:** 1.5
- **Dependencies:** T-05, T-06
- **Acceptance:** run 3+ ablation variants (no dropout, different lookback, different hidden size)
- **Author:** from-scratch
- **LOC est:** 40

## T-10: Extension Experiment
- **Status:** [ ] TODO
- **File:** experiments/
- **Est hrs:** 1.5
- **Dependencies:** T-05, T-06
- **Acceptance:** one of: hyperparam sweep 12+ trials, new dataset transfer, TFLite INT8 quantization comparison
- **Author:** from-scratch
- **LOC est:** 40

## T-11: Colab Notebook
- **Status:** [ ] TODO
- **File:** notebooks/research_colab.ipynb
- **Est hrs:** 1.0
- **Dependencies:** all above
- **Acceptance:** end-to-end train→eval→plot, smoke epoch=1 <45min on Colab Free
- **Author:** from-scratch (90% — Colab boilerplate 10%)
- **LOC est:** N/A (notebook)

## T-12: Documentation
- **Status:** [ ] TODO
- **File:** README.md, REPORT.md, REPRODUCE.md, CHANGELOG.md
- **Est hrs:** 2.0
- **Dependencies:** all above
- **Acceptance:** all 18 README sections, REPORT 4-8 pages, REPRODUCE complete
- **Author:** from-scratch
- **LOC est:** N/A (markdown)

## Summary

| Task | Status | LOC | Author Type |
|------|--------|-----|-------------|
| T-01 | DONE | 80 | from-scratch |
| T-02 | DONE | 40 | from-scratch |
| T-03 | DONE | 120 | from-scratch (95%) |
| T-04 | DONE | 20 | from-scratch |
| T-05 | DONE | 100 | from-scratch |
| T-06 | DONE | 60 | from-scratch (95%) |
| T-07 | DONE | 50 | from-scratch |
| T-08 | WIP | 40 | from-scratch (90%) |
| T-09 | TODO | 40 | from-scratch |
| T-10 | TODO | 40 | from-scratch |
| T-11 | TODO | N/A | from-scratch (90%) |
| T-12 | TODO | N/A | from-scratch |

**Running attribution %:** ~3% (TFLite API, sklearn API)
