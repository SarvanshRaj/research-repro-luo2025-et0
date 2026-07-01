# Demo Video Script — 2 Minutes

## 0:00–0:15 — Hook

**Voiceover:** "Precision agriculture needs forecasting models that run on $35 hardware. This is a from-scratch reproduction of a deep learning benchmark for ET0 prediction — open-source, Raspberry Pi deployable."

**On-screen:** Title card — "Edge ET0 Forecasting — Reproducible Benchmark"
**B-roll:** Terminal scrolling with training logs, sensor hardware close-up.

---

## 0:15–0:45 — Method

**Voiceover:** "I reimplemented five deep learning architectures — GRU, LSTM, RNN, TCN, and a lightweight Transformer — trained on meteorological data, and evaluated across three random seeds for statistical rigor."

**On-screen:** Architecture diagram fade-in — data pipeline → models → evaluation.
**B-roll:** Code snippets, model architecture ASCII diagram.

---

## 0:45–1:15 — Results

**Voiceover:** "The GRU model achieves R²=0.966 with just 40,000 parameters. TCN achieves similar accuracy with only 4,500 parameters and faster inference. Ablation shows the lookback window matters most — 12 hours beats 6 hours by nearly 1%."

**On-screen:** Results table — Paper vs Reproduction. Bar chart comparison.
**B-roll:** Terminal showing metrics, training curves plotting.

---

## 1:15–1:45 — Extension + Edge Demo

**Voiceover:** "A hyperparameter sweep identifies optimal training settings. The model exports to TFLite for Raspberry Pi deployment — sub-millisecond inference on ARM hardware. Complete pipeline: data loading, training, evaluation, export — all open-source."

**On-screen:** TFLite export process, latency measurement.
**B-roll:** Raspberry Pi board, terminal showing TFLite inference.

---

## 1:45–2:00 — Close

**Voiceover:** "Open-source, reproducible, Raspberry Pi-ready — useful for student labs, sustainability teams, and robotics builders worldwide. GitHub: github.com/SarvanshRaj/research-repro-luo2025-et0 — contributions welcome."

**On-screen:** GitHub URL, license badge, "Contributions Welcome"
**B-roll:** Terminal showing git clone, README preview.

---

## Production Notes

- Screen capture: training logs, architecture diagrams, terminal metrics
- Hardware: generic sensor close-up, Raspberry Pi board
- Tone: engineer-to-engineer, factual, no hype
- No university name-drops in close
- No personal B-roll — neutral professional
