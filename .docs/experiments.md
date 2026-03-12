# Experiment Log

## Experiment #0: Smoke Test
- **Date:** 2026-03-12
- **Type:** Verification run (3 steps only)
- **Changes:** None (default train.py)
- **Result:**
  - val_bpb: 2.206678
  - peak_vram_mb: 2985.3
  - tok/sec: ~78K
  - MFU: 10.5%
- **Status:** PASS — setup verified, all systems go
- **Notes:**
  - Autotune tested batch sizes 16, 8, 4 with checkpointing
  - Selected batch_size=8 as optimal (best throughput at 80K tok/sec)
  - VRAM usage very conservative (~3 GB / 16 GB available)

---

## Experiment #1: Baseline (full 5-min run)
- **Date:** 2026-03-12
- **Type:** Baseline establishment
- **Changes:** None (default train.py)
- **Result:** PENDING — running...
