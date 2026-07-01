# Attribution Log — External LOC Tracking

## Policy
- External (attributed) LOC must be ≤15% of total source code
- Core logic must be 100% authored from scratch
- API boilerplate (TFLite converter, PyTorch nn.Module base) counts as attributed

## Running Log

| Task | File | External LOC | License | Source URL | Running % |
|------|------|--------------|---------|------------|-----------|
| (initial) | — | 0 | — | — | 0% |

## Summary

- **Total authored LOC:** 0 (build not started)
- **External LOC:** 0
- **Attributed %:** 0%
- **Target:** ≤15%

## Notes

- TFLiteConverter API usage: standard boilerplate, not counted as core logic
- PyTorch nn.Module/nn.GRU/nn.LSTM: framework API, not counted
- sklearn metrics (r2_score, mean_absolute_error): API usage, not counted
- From-scratch TCN: dilated causal convolution — fully authored
- Data preprocessing/windowing: fully authored
- Training loop: fully authored
- Evaluation/bootstrap CI: fully authored
