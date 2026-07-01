# Runbook — Phase Checklist

## GATE-A
- [x] Search broad — Papers With Code, arXiv, Scholar
- [x] Score top 8 via weighted rubric
- [x] Select top 5 slate
- [x] Present GATE-A table
- [x] Await approval — APPROVED P-B

## GATE-B
- [x] 3-finalist + recommendation
- [x] Risk register + from-scratch plan + compute budget + SLO
- [x] Await approval — APPROVED P-B

## PHASE 0 — INIT
- [ ] Create directory scaffold
- [ ] git init + config
- [ ] Initial commit
- [ ] scratch.txt with messy notes
- [ ] tools_discovered.md seeded
- [ ] runbook.md (this file)
- [ ] Sign PHASE0_COMPLETE

## PHASE 1 — PAPER ACQUISITION
- [ ] Download/analyze paper
- [ ] BibTeX → paper/BIBTEX.bib
- [ ] notes/summary.md — full paper summary
- [ ] Sign PHASE1_COMPLETE

## PHASE 2 — TASK DECOMPOSITION
- [ ] notes/tasks.md — T-01 … T-N
- [ ] notes/attribution.md — LOC tracking
- [ ] Sign PHASE2_COMPLETE

## PHASE 3 — IMPLEMENTATION
- [ ] src/data.py — data loader
- [ ] src/model.py — GRU/LSTM/TCN/RNN
- [ ] src/train.py — training loop
- [ ] src/evaluate.py — metrics
- [ ] src/export_tflite.py — TFLite conversion
- [ ] src/utils.py — helpers
- [ ] notebooks/research_colab.ipynb
- [ ] requirements.txt + environment.yml
- [ ] LICENSE MIT + .gitignore
- [ ] Commit: v0.1-impl-complete
- [ ] Sign PHASE3_COMPLETE

## PHASE 4 — REPRODUCTION & VALIDATION
- [ ] Train n_seeds=3 (42, 123, 2026)
- [ ] experiments/results_original.csv
- [ ] experiments/metrics.md — paper vs repro table
- [ ] Ablation ≥3 variants
- [ ] experiments/results_ablation.csv
- [ ] Internal code audit
- [ ] notes/validation.md
- [ ] Sign PHASE4_COMPLETE PASS/FAIL

## PHASE 5 — EXTENSION
- [ ] Extension experiment chosen
- [ ] notes/extension.md
- [ ] experiments/results_extension.csv
- [ ] Figure PNG
- [ ] Sign PHASE5_COMPLETE

## PHASE 6 — DOCUMENTATION
- [ ] README.md — 18 sections
- [ ] REPORT.md — 4-8 pages
- [ ] REPRODUCE.md
- [ ] CHANGELOG.md
- [ ] docs/admissions_brief.md
- [ ] docs/demo_video_script.md
- [ ] Sign PHASE6_COMPLETE

## PHASE 7 — QUALITY GATES
- [ ] G1 Reproduction SLO
- [ ] G2 From-scratch provenance
- [ ] G3 Ablation
- [ ] G4 Extension
- [ ] G5 Documentation universal
- [ ] G6 Benchmarks measurable
- [ ] G7 Repo hygiene
- [ ] G8 Colab
- [ ] G9 Universal fit
- [ ] G10 GitHub ready
- [ ] G11 /GHOST audit
- [ ] Sign PHASE7_COMPLETE

## PHASE 8 — GITHUB PUBLISH
- [ ] Final commit
- [ ] Tag v1.0.0
- [ ] Push to github.com/SarvanshRaj/research-repro-luo2025-et0
- [ ] Verify README renders
- [ ] Sign PHASE8_COMPLETE

## PHASE 9 — FINAL REPORT
- [ ] operator_handoff.md
- [ ] REPORT.md executive summary
- [ ] Print structured terminal output
- [ ] Sign PHASE9_COMPLETE
