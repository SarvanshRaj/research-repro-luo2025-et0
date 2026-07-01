# Tools Discovered Log

| Tool | URL | Cost proof | Quota | License | Why chosen | Approved_at UTC | Risk |
|------|-----|------------|-------|---------|------------|-----------------|------|
| Google Colab Free | https://colab.research.google.com | Free tier, no credit card for basic | T4 GPU, ~12h sessions | Proprietary (Google) — free use | Seed tool — proven $0 GPU | 2026-07-01T10:00:00Z | Session disconnects |
| Kaggle Notebooks | https://www.kaggle.com/code | Free tier, no credit card | 30h GPU/week (P100/T4) | Proprietary (Google) — free use | Seed tool — backup GPU | 2026-07-01T10:00:00Z | Weekly quota limits |
| Hugging Face Spaces | https://huggingface.co/spaces | Free tier for CPU | CPU only, 2 vCPU, 16GB RAM | Proprietary (HF) — free use | Seed tool — demo hosting | 2026-07-01T10:00:00Z | No GPU on free |
| GitHub Actions | https://github.com/features/actions | Free for public repos | 2000 min/month | Proprietary (GitHub) — free use | CI/CD for tests | 2026-07-01T10:00:00Z | None for public |
| NASA POWER | https://power.larc.nasa.gov/ | Free, no auth | Unlimited queries | Open (NASA) | Alternative data source if Sichuan data unavailable | 2026-07-01T10:00:00Z | None |
| UrbanSound8K | https://urbansounddataset.weebly.com/urbansound8k.html | Free download | ~6GB | CC-BY (non-commercial research) | Backup dataset for extension | 2026-07-01T10:00:00Z | Non-commercial license |

---

## Notes

- Colab Free T4: 15GB RAM, ~12GB VRAM, sessions disconnect after ~12h idle
- Kaggle: 30h/week GPU, can be used as backup if Colab quota exhausted
- Paper data: Sichuan meteorological + soil sensor — verify download availability from Hydrology Research supplementary
