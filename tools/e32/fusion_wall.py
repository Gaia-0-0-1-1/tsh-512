"""fusion_wall.py — E32: the fusion-wall experiment (E25's deferred P3).

Can a network LEARN a composite function directly? 16 pairwise chain
compositions of the 4 banked tasks (the same specs E25 fingerprinted)
as training tasks; each trained from scratch; compared to the parts'
E20 grok rates.

Arms: 16 composites x 2 seeds (fp, d64, wd 0.5, 20k cap) = 32 runs
+ 4 parts x 2 seeds (the P3 replication control) = 8 runs.

Pre-registration: tsh-512 seq 196.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "e6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proto"))

from math_structures import STRUCTURES  # noqa: E402

TASKS = ["Z8", "Z4xZ2", "Z2x2x2", "Z7"]


def make_composite_task(outer, inner, seed=0):
    """The composed function as a training task.

    Input: ((a1,b1),(a2,b2)) — two pairs in inner's domain
    (values < min(n_outer, n_inner) for table safety).
    Label: outer(inner(a1,b1), inner(a2,b2)).
    """
    table_o = STRUCTURES[outer]["make"]()
    table_i = STRUCTURES[inner]["make"]()
    n = min(STRUCTURES[outer]["n"], STRUCTURES[inner]["n"])

    xs, ys = [], []
    for a1 in range(n):
        for b1 in range(n):
            for a2 in range(n):
                for b2 in range(n):
                    c1 = table_i[a1][b1]
                    c2 = table_i[a2][b2]
                    xs.append([a1, b1, a2, b2])
                    ys.append(table_o[c1 % STRUCTURES[outer]["n"]]
                              [c2 % STRUCTURES[outer]["n"]])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    rng = random.Random(seed)
    perm = rng.sample(range(len(xs)), len(xs))
    n_train = int(0.8 * len(xs))
    tr, te = perm[:n_train], perm[n_train:]
    return {"train_x": x[tr], "train_y": y[tr],
            "test_x": x[te], "test_y": y[te],
            "vocab": n, "out_vocab": STRUCTURES[outer]["n"]}


def make_part_task(task, seed=0):
    """A component task at the same config (the P3 control)."""
    spec = STRUCTURES[task]
    table = spec["make"]()
    n = spec["n"]
    xs, ys = [], []
    for a in range(n):
        for b in range(n):
            xs.append([a, b])
            ys.append(table[a][b])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    rng = random.Random(seed)
    perm = rng.sample(range(len(xs)), len(xs))
    n_train = int(0.8 * len(xs))
    tr, te = perm[:n_train], perm[n_train:]
    return {"train_x": x[tr], "train_y": y[tr],
            "test_x": x[te], "test_y": y[te],
            "vocab": n, "out_vocab": n}


class Tiny(nn.Module):
    def __init__(self, vocab, out_vocab, d=64, seq_len=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.w_in = nn.Linear(d, 4 * d)
        self.w_out = nn.Linear(4 * d, d)
        self.unembed = nn.Linear(d, out_vocab, bias=False)
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


def run_task(ds, seed, max_steps=20000):
    torch.manual_seed(seed)
    model = Tiny(ds["vocab"], ds["out_vocab"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=0.5, betas=(0.9, 0.98))
    x, y = ds["train_x"], ds["train_y"]
    grok_step = None
    te = 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randperm(x.shape[0])[:min(64, x.shape[0])]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                te = (model(ds["test_x"]).argmax(-1)
                          == ds["test_y"]).float().mean().item()
            if te >= 0.95:
                grok_step = step
                break
    return {"grok_step": grok_step, "final_test": te, "steps": step}


def main():
    print("E32: THE FUSION WALL — can composites be learned directly?\n")
    results = []

    # P3: the parts control
    print("--- P3: parts (the replication control) ---")
    for task in TASKS:
        for seed in (0, 1):
            ds = make_part_task(task, seed)
            r = run_task(ds, seed)
            grok = r["grok_step"] if r["grok_step"] else "never"
            print(f"  {task:<10} s{seed}: grok={grok} "
                  f"test={r['final_test']:.3f}")
            results.append({"kind": "part", "task": task, "seed": seed, **r})

    # the composites
    print("\n--- the 16 composites ---")
    for outer in TASKS:
        for inner in TASKS:
            for seed in (0, 1):
                ds = make_composite_task(outer, inner, seed)
                r = run_task(ds, seed)
                grok = r["grok_step"] if r["grok_step"] else "never"
                print(f"  {outer}({inner}) s{seed}: grok={grok} "
                      f"test={r['final_test']:.3f}")
                results.append({"kind": "composite", "outer": outer,
                                "inner": inner, "seed": seed, **r})

    out = Path(__file__).parent / "results.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # analysis
    print("\n=== ANALYSIS ===\n")
    parts = [r for r in results if r["kind"] == "part"]
    comps = [r for r in results if r["kind"] == "composite"]
    parts_grok = sum(1 for r in parts if r["grok_step"])
    comps_grok = sum(1 for r in comps if r["grok_step"])
    print(f"  parts grok rate: {parts_grok}/{len(parts)}")
    print(f"  composite grok rate: {comps_grok}/{len(comps)}")
    print()

    # P2: the asymmetry
    easy_inner = [r for r in comps if r["inner"] == "Z2x2x2"]
    easy_outer = [r for r in comps if r["outer"] == "Z2x2x2"]
    ei_grok = sum(1 for r in easy_inner if r["grok_step"])
    eo_grok = sum(1 for r in easy_outer if r["grok_step"])
    print(f"  Z2x2x2 as INNER: {ei_grok}/{len(easy_inner)} grok")
    print(f"  Z2x2x2 as OUTER: {eo_grok}/{len(easy_outer)} grok")

    print(f"\n=== VERDICTS ===")
    rate = comps_grok / len(comps)
    if rate <= 0.5:
        print(f"  P1 (fusion wall exists): CONFIRMED — "
              f"only {comps_grok}/{len(comps)} composites grok")
    else:
        print(f"  P1 (fusion wall exists): FALSIFIED — "
              f"{comps_grok}/{len(comps)} composites grok (the free "
              f"monoid extends to learning)")
    if ei_grok != eo_grok:
        better = "INNER" if ei_grok > eo_grok else "OUTER"
        print(f"  P2 (asymmetry): {better} placement matters "
              f"({ei_grok} vs {eo_grok})")
    else:
        print(f"  P2 (asymmetry): no difference ({ei_grok} vs {eo_grok})")
    print(f"  P3 (parts replicate): "
          f"{'CONFIRMED' if parts_grok >= 4 else 'CHECK'} "
          f"({parts_grok}/{len(parts)})")


if __name__ == "__main__":
    main()
