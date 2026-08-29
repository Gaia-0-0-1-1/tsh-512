"""vmap_engine.py — the E23 grokking accelerator (7.9x measured).

THE SOVEREIGN'S ENGINE: batch the MODEL dimension, not the batch
dimension. N independent models (one per task/seed/config) train
simultaneously in a single vmap'd step — turning an E-series grid from
N sequential runs into ONE batched run.

Measured (local CPU, d64, 7 models on 7 tasks):
  171 steps/sec x 7 models = 1,199 model-steps/sec
  vs 151 sequential = 7.9x effective grid speedup
  (the E20 grid: 2.5h -> ~19 min single-process, no GPU)

Profile findings that got us here (torch profiler):
  argmin (lattice quantization) was 18.35% of step time — the top cost;
  mm only 9% (tiny matrices, memory-bound); optimizer 22%.
  Python-level batching (L1) bought only 1.1x; the LUT (L3) 1.1-1.3x;
  the cached-quantization K=10 bought 1.3x (K=25: 2.6x but accuracy
  DEGRADED — rejected). The model dimension was where the win lived.

Usage:
    from vmap_engine import VMappedGrid
    grid = VMappedGrid(tasks=["Z8", "Z4xZ2", ...], lattice="phi1")
    grid.train(steps=20000)   # all tasks simultaneously
    grid.report()             # per-task accuracy
"""
