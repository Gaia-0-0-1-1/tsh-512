"""wd_relief.py — E47: the wd-relief sweep on composites.

E46 opened the T7 phoneme gate at wd=1.0. Does the same relief
apply to the composite near-misses — and to the hard three-factor
cells? This experiment draws the line between config-margin
coin-flips and the config-robust wall.

Cells (4 seeds each, wd=1.0, 50k cap, the E32 protocol):
  near-misses: Z6(Z6), MUL8(MUL8), T8(T8), PD(PD)
  hard E32:    Z4xZ2(Z4xZ2), Z2x2x2(Z2x2x2),
               Z4xZ2(Z2x2x2), Z2x2x2(Z4xZ2)

Pre-registration: tsh-512 seq 237.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TSH = HERE.parents[1]

sys.path.insert(0, str(TSH / "tools" / "e6"))
sys.path.insert(0, str(TSH / "proto"))
sys.path.insert(0, str(TSH / "tools" / "e20"))

from math_structures import STRUCTURES  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402

RESULTS = []


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def log(kind, **kw):
    rec = {"kind": kind, **kw}
    RESULTS.append(rec)
    print(f"  [logged] {kind}: " +
          " ".join(f"{k}={v}" for k, v in kw.items()))


def save():
    with open(HERE / "results.jsonl", "w", encoding="utf-8") as f:
        for r in RESULTS:
            f.write(canon(r) + "\n")


def cyc(n, domain=8):
    return [[(a + b) % n for b in range(domain)] for a in range(domain)]


def make_MUL8():
    return [[(a * b) % 8 for b in range(8)] for a in range(8)]


PERM = [3, 0, 7, 4, 1, 6, 5, 2]


def make_PD():
    z8 = cyc(8)
    return [[PERM[v] for v in row] for row in z8]


def composite_truth(outer_t, inner_t, n):
    """E32's convention at general domain size n (both tables
    indexed 0..n-1 in each arg; vocab values may wrap)."""
    xs, ys = [], []
    for a1 in range(n):
        for b1 in range(n):
            for a2 in range(n):
                for b2 in range(n):
                    c1 = inner_t[a1][b1]
                    c2 = inner_t[a2][b2]
                    xs.append([a1, b1, a2, b2])
                    ys.append(outer_t[c1][c2])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def train_composite(outer_t, inner_t, n, vocab, seed=0, wd=1.0,
                    max_steps=50000):
    x, y = composite_truth(outer_t, inner_t, n)
    rng = random.Random(seed)
    perm = rng.sample(range(len(x)), len(x))
    tr, te = perm[:int(0.8 * len(x))], perm[int(0.8 * len(x)):]
    torch.manual_seed(seed)
    model = TinyTransformer(vocab, vocab, d=64, lattice=None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=wd, betas=(0.9, 0.98))
    x_tr, y_tr = x[tr], y[tr]
    x_te, y_te = x[te], y[te]
    grok_step, te_acc = None, 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randperm(x_tr.shape[0])[:64]
        loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                te_acc = (model(x_te).argmax(-1)
                          == y_te).float().mean().item()
            if te_acc >= 0.95:
                grok_step = step
                break
    return {"grok_step": grok_step, "final_test": round(te_acc, 4)}


def main():
    t0 = time.time()
    print("E47: THE WD-RELIEF SWEEP — coin-flips vs the config-robust"
          " wall")
    print("pre-registered seq 237\n")
    print("config: wd=1.0, 50k cap, 4 seeds per family\n")

    tables = {
        "Z6": cyc(6), "MUL8": make_MUL8(), "T8": cyc(8),
        "PD": make_PD(),
        "Z4xZ2": STRUCTURES["Z4xZ2"]["make"](),
        "Z2x2x2": STRUCTURES["Z2x2x2"]["make"](),
    }
    families = [
        # (name, outer, inner, domain_n, vocab)
        ("Z6(Z6)", "Z6", "Z6", 6, 6),
        ("MUL8(MUL8)", "MUL8", "MUL8", 8, 8),
        ("T8(T8)", "T8", "T8", 8, 8),
        ("PD(PD)", "PD", "PD", 8, 8),
        ("Z4xZ2(Z4xZ2)", "Z4xZ2", "Z4xZ2", 8, 8),
        ("Z2x2x2(Z2x2x2)", "Z2x2x2", "Z2x2x2", 8, 8),
        ("Z4xZ2(Z2x2x2)", "Z4xZ2", "Z2x2x2", 8, 8),
        ("Z2x2x2(Z4xZ2)", "Z2x2x2", "Z4xZ2", 8, 8),
    ]

    fam_results = {}
    for name, outer, inner, n, vocab in families:
        grokked = 0
        for seed in range(4):
            r = train_composite(tables[outer], tables[inner], n,
                                vocab, seed=seed)
            ok = r["grok_step"] is not None
            grokked += ok
            g = r["grok_step"] if ok else "never"
            print(f"  {name} s{seed}: grok={g} "
                  f"test={r['final_test']}")
            log("composite", family=name, seed=seed, **r)
        fam_results[name] = grokked
        print(f"  -> {name}: {grokked}/4 at wd=1.0\n")
        save()

    print("=== VERDICTS (pre-registered seq 237) ===")
    near = {k: fam_results[k] for k in
            ("Z6(Z6)", "MUL8(MUL8)", "T8(T8)")}
    p1 = all(v >= 3 for v in near.values())
    print(f"  P1 (near-misses rescue at wd=1.0): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} {near}")
    p2 = fam_results["PD(PD)"] >= 2
    print(f"  P2 (PD(PD) opens — representation wall is "
          f"config-relative): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} "
          f"({fam_results['PD(PD)']}/4)")
    hard = {k: fam_results[k] for k in
            ("Z4xZ2(Z4xZ2)", "Z2x2x2(Z2x2x2)",
             "Z4xZ2(Z2x2x2)", "Z2x2x2(Z4xZ2)")}
    p3 = all(v <= 1 for v in hard.values())
    print(f"  P3 (hard cells stay walled at wd=1.0): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} {hard}")
    log("verdicts", P1_near_rescue=p1, P2_pd_opens=p2,
        P3_hard_walled=p3, fam=fam_results)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
