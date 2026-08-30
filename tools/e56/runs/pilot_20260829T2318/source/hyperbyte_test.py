"""hyperbyte_test.py — E20: the P-HYPERBYTE CPU test.

The architectural falsification: does per-unit lattice selection (the
hyperbyte's base_id made trainable) beat the best fixed lattice?

Arms (matched d64 one-layer transformer, E7 task battery):
  fp       — full precision (the baseline)
  ternary  — all matrices at the ternary lattice {-1,0,+1} x gamma
  phi1     — all matrices at the 5-state golden lattice
  phi2     — all matrices at the 9-state Stakhov lattice
  hyper    — PER-MATRIX lattice: each linear layer carries a learnable
             categorical router over {ternary, phi1, phi2} (the
             hyperbyte base_id, trained by Gumbel-softmax)

Batched execution: all 7 tasks run in a single tensor (Level 1
optimization) — the whole grid targets < 2 hours on local CPU.

Pre-registration: tsh-512 seq 160.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "e6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proto"))
from math_structures import STRUCTURES, make_task  # noqa: E402

PHI = (1 + math.sqrt(5)) / 2
# lattice state sets (scaled by gamma = mean|W|)
LATTICES = {
    "ternary": [0.0, -1.0, 1.0],
    "phi1": [0.0, -1.0, 1.0, -PHI, PHI],
    "phi2": [0.0, -1.0, 1.0, -1 / PHI, 1 / PHI, -PHI, PHI, -PHI * PHI, PHI * PHI],
}


class LatticeLinear(nn.Module):
    """Linear layer with a lattice-quantized effective weight.

    lattice=None -> plain fp linear (baseline)
    lattice=str  -> STE quantization to that lattice
    lattice='hyper' -> learnable per-layer router over the lattices
                       (the hyperbyte base_id, Gumbel-softmax trained)
    """

    def __init__(self, in_f, out_f, lattice=None, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.lattice = lattice
        if lattice == "hyper":
            # the router logits: one per lattice choice
            self.router_logits = nn.Parameter(torch.zeros(len(LATTICES)))
            # cache the state tables
            self._tables = [torch.tensor(LATTICES[k]) for k in
                            ("ternary", "phi1", "phi2")]

    def effective_weight(self):
        w = self.weight.detach()
        if self.lattice is None or self.lattice == "fp":
            return w
        gamma = w.abs().mean().clamp_min(1e-8)
        xn = w / gamma
        if self.lattice == "hyper":
            # Gumbel-softmax over lattice choices (differentiable)
            probs = F.gumbel_softmax(self.router_logits, tau=1.0, hard=False)
            # quantize to each lattice, mix by probs
            q_mixed = torch.zeros_like(xn)
            for i, table in enumerate(self._tables):
                idx = self._nearest(xn, table)
                q_mixed = q_mixed + probs[i] * table[idx]
            return q_mixed * gamma
        table = torch.tensor(LATTICES[self.lattice], device=w.device)
        idx = self._nearest(xn, table)
        return table[idx] * gamma

    @staticmethod
    def _nearest(xn, table):
        # xn: (out, in), table: (k,) -> index tensor (out, in)
        dist = (xn.unsqueeze(-1) - table.reshape(1, 1, -1)).abs()
        return dist.argmin(-1)

    def forward(self, x):
        if self.lattice is None or self.lattice == "fp":
            return F.linear(x, self.weight, self.bias)
        w_eff = self.effective_weight()
        w = self.weight + (w_eff - self.weight).detach()  # STE
        return F.linear(x, w, self.bias)


class TinyTransformer(nn.Module):
    def __init__(self, vocab, out_vocab, d=64, lattice=None):
        super().__init__()
        def lin(i, o):
            return LatticeLinear(i, o, lattice)
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, 4, d))
        self.q = lin(d, d)
        self.k = lin(d, d)
        self.v = lin(d, d)
        self.o = lin(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.w_in = lin(d, 4 * d)
        self.w_out = lin(4 * d, d)
        self.unembed = lin(d, out_vocab)
        self.d = d

    def forward(self, x):
        B, L = x.shape
        e = self.embed(x) + self.pos[:, :L]
        z = self.ln1(e)
        q, k, v = self.q(z), self.k(z), self.v(z)
        att = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        att = att.softmax(-1)
        h = e + self.o(att @ v)
        h = h + self.w_out(F.relu(self.w_in(self.ln2(h))))
        return self.unembed(h.mean(dim=1))


def run_arm(arm, task_name, seed, max_steps=20000, eval_every=200,
            device="cpu"):
    """Train one (arm, task, seed) cell. Returns summary dict."""
    spec = STRUCTURES[task_name]
    ds = make_task(task_name, 0.8, seed, device)
    n = spec["n"]

    torch.manual_seed(seed)
    model = TinyTransformer(n, n, d=64, lattice=arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.5,
                            betas=(0.9, 0.98))

    memorize_step = None
    grok_step = None
    t0 = time.time()

    for step in range(1, max_steps + 1):
        idx = torch.randperm(ds["train_x"].shape[0])[:min(64, ds["train_x"].shape[0])]
        logits = model(ds["train_x"][idx])
        loss = F.cross_entropy(logits, ds["train_y"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0:
            with torch.no_grad():
                tr = (model(ds["train_x"]).argmax(-1) == ds["train_y"]).float().mean().item()
                te = (model(ds["test_x"]).argmax(-1) == ds["test_y"]).float().mean().item()
            if memorize_step is None and tr >= 0.99:
                memorize_step = step
            if grok_step is None and te >= 0.95:
                grok_step = step
                break  # early stop on grok — the metric is the step count

    return {
        "arm": arm, "task": task_name, "seed": seed,
        "memorize_step": memorize_step, "grok_step": grok_step,
        "steps_run": step, "final_test": te,
        "secs": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fp,ternary,phi1,phi2,hyper")
    ap.add_argument("--tasks", default="Z8,Z4xZ2,Z2x2x2,Q8,D4,S3,Z7")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--out", default="runs/e20_hyperbyte/results.jsonl")
    args = ap.parse_args()

    arms = args.arms.split(",")
    tasks = args.tasks.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(out, "a")

    total = len(arms) * len(tasks) * len(seeds)
    print(f"E20 hyperbyte test: {len(arms)} arms x {len(tasks)} tasks x "
          f"{len(seeds)} seeds = {total} runs")
    print(f"arms: {arms}\n")

    done = 0
    t_start = time.time()
    for arm in arms:
        for task in tasks:
            for seed in seeds:
                rec = run_arm(arm, task, seed, args.max_steps)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                done += 1
                grok = rec["grok_step"] if rec["grok_step"] else "never"
                print(f"  [{done}/{total}] {arm:<8} {task:<8} s{seed}: "
                      f"grok={grok} test={rec['final_test']:.3f} "
                      f"({rec['secs']}s)", flush=True)
    log.close()
    print(f"\ntotal: {time.time() - t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
