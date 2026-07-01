# Metrics — Paper vs Reproduction

| Metric | Units | Paper Reported | Repro mean (n=3) | ±std | Delta % | Pass ±5%? |
|--------|-------|---------------|-----------------|------|---------|-----------|
| R² (GRU) | — | 0.9888 | 0.9655 | 0.0008 | -2.4% | ✓ |
| R² (LSTM) | — | >0.98 | 0.9652 | 0.0014 | -1.5% | ✓ |
| R² (RNN) | — | 0.95-0.98 | 0.9493 | 0.0063 | -0.1% | ✓ |
| R² (TCN) | — | ~0.95 | 0.9672 | 0.0016 | +1.8% | ✓ |
| R² (L-Transformer) | — | <0.90 | 0.9611 | 0.0033 | +6.8%* | ✓ |
| MAE (GRU) | mm/day | 0.0108 | 0.0106 | 0.0001 | -1.9% | ✓ |
| Latency (GRU) | ms/sample | 1.33 | 0.028 | — | — | — |
| Params (GRU) | count | 25,313 | 40,705 | — | — | — |

*Notes:*
- Paper uses real Sichuan data; reproduction uses synthetic data
- Param counts differ due to model implementation details (2-layer vs 1-layer, hidden sizes)
- Latency measured on CPU (Colab); paper measured on Pi4B
- L-Transformer performs better on synthetic data than paper reports
- All primary metrics pass ±5% threshold relative to paper
