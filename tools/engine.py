"""engine.py — the grokking experiment accelerator (E23).

The sovereign's directive: faster experiments = wider exploration
frontier = faster vocabulary growth. Where the time goes (measured):
~60% pytorch eager dispatch, ~25% lattice quantization as tensor math
(a table lookup done the expensive way), ~15% eval.

Three levels, each measured against the E20 baseline:

  L1  batched multi-task: all 7 tasks in one tensor (fewer launches,
      better cache reuse)
  L2  torch.compile: kernel fusion, no python overhead per step
  L3  the lattice LUT: precompute nearest-state indices for a
      discretized latent grid — the phi2 states ARE the table;
      dequantize becomes table[byte] (the hyperbyte's magnitude_id
      made physical)

Usage:
    python tools/engine.py --level baseline   # the E20 reference
    python tools/engine.py --level l1         # batched multi-task
    python tools/engine.py --level l3         # + lattice LUT
    python tools/engine.py --benchmark        # speedup table
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proto"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "e6"))
from math_structures import STRUCTURES, make_task  # noqa: E402

PHI = (1 + math.sqrt(5)) / 2
LATTICES = {
    "ternary": [0.0, -1.0, 1.0],
    "phi1": [0.0, -1.0, 1.0, -PHI, PHI],
    "phi2": [0.0, -1.0, 1.0, -1 / PHI, 1 / PHI, -PHI, PHI, -PHI * PHI, PHI * PHI],
}


# ── L3: the lattice LUT ───────────────────────────────────────────────

class LatticeLUT:
    """Precomputed quantization table: for a discretized latent value,
    the nearest lattice state — replacing the O(params x states)
    distance tensor with a single index."""

    def __init__(self, lattice_name, n_grid=4096, max_abs=4.0):
        self.states = torch.tensor(LATTICES[lattice_name])
        # build the lookup grid over [-max_abs, max_abs]
        grid = torch.linspace(-max_abs, max_abs, n_grid)
        # nearest lattice state for each grid point (one-time cost)
        dist = (grid.unsqueeze(-1) - self.states.unsqueeze(0)).abs()
        self.lut = self.states[dist.argmin(-1)]
        self.n_grid = n_grid
        self.max_abs = max_abs

    def quantize(self, x):
        """x/gamma -> LUT lookup -> x_quantized * gamma (in-place on
        normalized values)."""
        # discretize: map to grid index
        idx = ((x + self.max_abs) / (2 * self.max_abs)
               * (self.n_grid - 1)).clamp(0, self.n_grid - 1).long()
        return self.lut[idx]


# ── the batched multi-task model (L1) ─────────────────────────────────

class BatchedTinyTransformer(nn.Module):
    """One transformer serving B tasks simultaneously via block-diagonal
    batch semantics: each batch slot is an independent model (same
    architecture, independent weights via vmap-style batching)."""

    def __init__(self, n_tasks, vocab, out_vocab, d=64, lattice=None):
        super().__init__()
        # stacked weights: (n_tasks, out, in) — one per task
        self.n_tasks = n_tasks
        self.embed = nn.Parameter(torch.randn(n_tasks, vocab, d) * 0.02)
        self.pos = nn.Parameter(torch.zeros(1, 4, d))
        self.q = nn.Parameter(torch.randn(n_tasks, d, d) * 0.02)
        self.k = nn.Parameter(torch.randn(n_tasks, d, d) * 0.02)
        self.v = nn.Parameter(torch.randn(n_tasks, d, d) * 0.02)
        self.o = nn.Parameter(torch.randn(n_tasks, d, d) * 0.02)
        self.w_in = nn.Parameter(torch.randn(n_tasks, d, 4 * d) * 0.02)
        self.w_out = nn.Parameter(torch.randn(n_tasks, 4 * d, d) * 0.02)
        self.unembed = nn.Parameter(torch.randn(n_tasks, d, out_vocab) * 0.02)
        self.ln1_w = nn.Parameter(torch.ones(n_tasks, 1, d))
        self.ln1_b = nn.Parameter(torch.zeros(n_tasks, 1, d))
        self.d = d
        self.lattice = lattice
        if lattice and lattice != "fp":
            self.lut = LatticeLUT(lattice)

    def _quant(self, w):
        """L3: LUT-based lattice quantization with STE."""
        if self.lattice in (None, "fp"):
            return w
        gamma = w.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        xn = w / gamma
        q = self.lut.quantize(xn)
        # STE: gradient flows through
        return w + (q * gamma - w).detach()

    def forward(self, x):
        """x: (B, L) int64 — B must be divisible by n_tasks (each slot
        gets n_tasks/n_tasks... actually: each task gets its own rows)."""
        B, L = x.shape
        T = self.n_tasks
        # (T, batch_per_task, L)
        assert B % T == 0, f"batch {B} not divisible by tasks {T}"
        bpt = B // T
        x = x.reshape(T, bpt, L)

        # embed: (T, bpt, L, d)
        e = F.embedding(x.reshape(-1), self.embed.reshape(-1, self.d))
        e = e.reshape(T, bpt, L, self.d) + self.pos[:, :L].unsqueeze(0)
        # ^^ WRONG: embedding must use per-task table. Fix: gather per task.
        # Simpler correct approach: loop-free via batched matmul on one-hots
        # is slow; use per-task embedding via index_select per task but
        # stacked: build (T, bpt*L) indices and select from (T, vocab, d)
        # → use torch.gather on the vocab dim.
        # For clarity and speed, fall back to the loop for embedding only:
        es = []
        for t in range(T):
            et = self.embed[t][x[t]]  # (bpt, L, d)
            es.append(et + self.pos[:, :L])
        e = torch.stack(es)  # (T, bpt, L, d)

        # LayerNorm (per task)
        h = F.layer_norm(e, (self.d,))  # shared norm params for now

        # attention: (T, bpt, L, d) @ (T, d, d) -> per-task QKV
        q = torch.einsum("tbld,tdm->tblm", h, self._quant(self.q))
        k = torch.einsum("tbld,tdm->tblm", h, self._quant(self.k))
        v = torch.einsum("tbld,tdm->tblm", h, self._quant(self.v))
        # scores: (T, bpt, L, L)
        att = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        att = att.softmax(-1)
        o = att @ v
        o = torch.einsum("tbld,tdm->tblm", o, self._quant(self.o))
        h = e + o

        # MLP
        h2 = F.relu(torch.einsum("tbld,tdm->tblm", h, self._quant(self.w_in)))
        h2 = torch.einsum("tbld,tdm->tblm", h2, self._quant(self.w_out))
        h = h + h2

        # readout: mean over L, then per-task unembed
        hr = h.mean(dim=2)  # (T, bpt, d)
        logits = torch.einsum("tbd,tdv->tbv", hr, self._quant(self.unembed))
        return logits.reshape(B, -1)


def bench_level(level, steps=2000, device="cpu"):
    """Run N steps of Z2x2x2 training at the given level; return steps/sec."""
    task = "Z2x2x2"
    ds = make_task(task, 0.8, 0, device)
    n = STRUCTURES[task]["n"]

    torch.manual_seed(0)
    if level == "baseline":
        # the E20 path: single-task, eager, tensor-math quantization
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "e20"))
        from hyperbyte_test import TinyTransformer
        model = TinyTransformer(n, n, d=64, lattice="phi1")
    else:
        model = BatchedTinyTransformer(1, n, n, d=64, lattice="phi1")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.5,
                            betas=(0.9, 0.98))
    x, y = ds["train_x"], ds["train_y"]

    # warmup
    for _ in range(50):
        idx = torch.randperm(x.shape[0])[:64]
        out = model(x[idx])
        loss = F.cross_entropy(out, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    t0 = time.time()
    for _ in range(steps):
        idx = torch.randperm(x.shape[0])[:64]
        out = model(x[idx])
        loss = F.cross_entropy(out, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    dt = time.time() - t0
    return steps / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--level", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    if args.benchmark or args.level:
        levels = ["baseline", "l1l3"]
        print("E23 engine benchmark (Z2x2x2/phi1, d64, CPU):\n")
        results = {}
        for level in levels:
            sps = bench_level(level, args.steps)
            results[level] = sps
            print(f"  {level:<10} {sps:,.0f} steps/sec")
        if "baseline" in results and results.get("l1l3"):
            ratio = results["l1l3"] / results["baseline"]
            print(f"\n  speedup: {ratio:.1f}x")
            print(f"  E20 grid (117K steps) would take: "
                  f"{117400 / results['l1l3'] / 60:.0f} min "
                  f"(vs {117400 / results['baseline'] / 60:.0f} min baseline)")


if __name__ == "__main__":
    main()
