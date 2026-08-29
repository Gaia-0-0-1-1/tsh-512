"""cross_grid.py — E44: cross-structural mismatch — the third factor?

E43's two-factor law (alignment x entropy) explained self-composites.
E32's hardest walls were CROSS-structural (0/16 for Z4xZ2-outer and
Z2x2x2-outer families). Is mismatch an independent third factor, or
alignment in disguise?

Arms (4 seeds each, the E32/E43 config):
  1. the cyclic-cross grid: T_k(T_j), k,j in {5,6,7,8}, k != j
  2. the scrambled-cross control: PD(T8), T8(PD)
  3. the E32 replication cells: Z4xZ2(Z2x2x2), Z2x2x2(Z4xZ2)

Pre-registration: tsh-512 seq 229.
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


def make_T(k):
    return [[(a + b) % k for b in range(8)] for a in range(8)]


PERM = [3, 0, 7, 4, 1, 6, 5, 2]  # E43's fixed scramble


def make_PD():
    z8 = make_T(8)
    return [[PERM[v] for v in row] for row in z8]


def composite_truth(outer_t, inner_t):
    """outer(inner(p1), inner(p2)) — the E32 convention, vocab 8."""
    xs, ys = [], []
    for a1 in range(8):
        for b1 in range(8):
            for a2 in range(8):
                for b2 in range(8):
                    c1 = inner_t[a1][b1]
                    c2 = inner_t[a2][b2]
                    xs.append([a1, b1, a2, b2])
                    ys.append(outer_t[c1][c2])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def train_composite(outer_t, inner_t, seed=0, max_steps=20000):
    x, y = composite_truth(outer_t, inner_t)
    rng = random.Random(seed)
    perm = rng.sample(range(len(x)), len(x))
    tr, te = perm[:int(0.8 * len(x))], perm[int(0.8 * len(x)):]
    torch.manual_seed(seed)
    model = TinyTransformer(8, 8, d=64, lattice=None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=0.5, betas=(0.9, 0.98))
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


def run_family(name, outer_t, inner_t, n_seeds=4):
    grokked, steps = 0, []
    for seed in range(n_seeds):
        r = train_composite(outer_t, inner_t, seed=seed)
        ok = r["grok_step"] is not None
        grokked += ok
        steps.append(r["grok_step"])
        g = r["grok_step"] if ok else "never"
        print(f"  {name} s{seed}: grok={g} test={r['final_test']}")
        log("composite", family=name, seed=seed, **r)
    return grokked


def main():
    t0 = time.time()
    print("E44: CROSS-STRUCTURAL MISMATCH — the third factor?")
    print("pre-registered seq 229\n")

    tables = {f"T{k}": make_T(k) for k in (5, 6, 7, 8)}
    tables["PD"] = make_PD()
    tables["Z4xZ2"] = STRUCTURES["Z4xZ2"]["make"]()
    tables["Z2x2x2"] = STRUCTURES["Z2x2x2"]["make"]()

    # ── arm 1: the cyclic-cross grid ──
    print("=== arm 1: the cyclic-cross grid T_k(T_j), k != j ===")
    grid = {}
    for k in (5, 6, 7, 8):
        for j in (5, 6, 7, 8):
            if k == j:
                continue
            name = f"T{k}(T{j})"
            print(f"  --- {name} ---")
            grid[name] = run_family(name, tables[f"T{k}"],
                                    tables[f"T{j}"])
            save()

    # ── arm 2: the scrambled-cross control ──
    print("\n=== arm 2: scrambled-cross — PD(T8) vs T8(PD) ===")
    pd_outer = run_family("PD(T8)", tables["PD"], tables["T8"])
    pd_inner = run_family("T8(PD)", tables["T8"], tables["PD"])
    save()

    # ── arm 3: the E32 replication cells ──
    print("\n=== arm 3: the E32 replication cells ===")
    rep1 = run_family("Z4xZ2(Z2x2x2)", tables["Z4xZ2"],
                      tables["Z2x2x2"])
    rep2 = run_family("Z2x2x2(Z4xZ2)", tables["Z2x2x2"],
                      tables["Z4xZ2"])
    save()

    # ── verdicts ──
    print("\n=== VERDICTS (pre-registered seq 229) ===")
    print(f"  cyclic-cross grid: {grid}")
    low_cells = {k: v for k, v in grid.items() if v <= 2}
    p1 = len(low_cells) >= 4
    print(f"  P1 (mismatch is real, >=4 cross cells <=2/4): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"({len(low_cells)} low cells of 12)")
    p2 = pd_outer < pd_inner
    print(f"  P2 (outer alignment matters more: PD(T8)={pd_outer} "
          f"< T8(PD)={pd_inner}): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'}")
    p3 = rep1 == 0 and rep2 == 0
    print(f"  P3 (E32 replication: both cross cells 0/4): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(Z4xZ2(Z2x2x2)={rep1}/4, Z2x2x2(Z4xZ2)={rep2}/4)")
    log("verdicts", P1_mismatch=p1, P2_outer_alignment=p2,
        P3_replication=p3, grid=grid,
        PD_T8=pd_outer, T8_PD=pd_inner,
        Z4Z2_cross=rep1, Z2x_cross=rep2,
        low_cells=len(low_cells))
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
