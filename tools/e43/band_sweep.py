"""band_sweep.py — E43: precision sweep of the entropy law.

Dense cells across the coin-flip band [2.3, 3.0] bits at fixed
vocab 8, with 4 seeds per composite (the band needs seed
statistics):

  T_k(a,b) = (a+b) mod k   k = 5, 6, 7, 8
    label entropy: ~2.29, ~2.58, ~2.75, 3.0
  PD_j: Z8 with a scrambled output permutation (entropy 3.0,
    encoding changed) — the invariance check

Pre-registration: tsh-512 seq 226.
"""
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TSH = HERE.parents[1]

sys.path.insert(0, str(TSH / "tools" / "e6"))
sys.path.insert(0, str(TSH / "proto"))
sys.path.insert(0, str(TSH / "tools" / "e20"))

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


def make_PD(perm):
    """Z8 with outputs permuted by pi (entropy unchanged at 3.0)."""
    z8 = [[(a + b) % 8 for b in range(8)] for a in range(8)]
    return [[perm[v] for v in row] for row in z8]


def entropy_bits(counter):
    total = sum(counter.values())
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log2(p)
    return round(h, 4)


def table_entropy(table):
    return entropy_bits(Counter(v for row in table for v in row))


def composite_label_entropy(t):
    """H of the self-composite's labels over the 4096-point domain."""
    dist = Counter()
    for a1 in range(8):
        for b1 in range(8):
            for a2 in range(8):
                for b2 in range(8):
                    dist[t[t[a1][b1]][t[a2][b2]]] += 1
    return entropy_bits(dist)


# ── trainers (the E32/E40 protocol, general) ─────────────────────────

def grok_phoneme(table, name, seed=0, max_steps=20000):
    xs = [[a, b] for a in range(8) for b in range(8)]
    ys = [table[a][b] for a in range(8) for b in range(8)]
    fx = torch.tensor(xs, dtype=torch.long)
    fy = torch.tensor(ys, dtype=torch.long)
    for attempt in range(6):
        g = torch.Generator().manual_seed(seed + attempt)
        perm = torch.randperm(64, generator=g)
        tr, te = perm[:51], perm[51:]
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(8, 8, d=64, lattice="phi1")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        x_tr, y_tr = fx[tr], fy[tr]
        grok_step, te_acc = None, 0.0
        for step in range(1, max_steps + 1):
            idx = torch.randperm(x_tr.shape[0])[:51]
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                with torch.no_grad():
                    te_acc = (model(fx[te]).argmax(-1)
                              == fy[te]).float().mean().item()
                if te_acc >= 0.95:
                    grok_step = step
                    break
        model.eval()
        with torch.no_grad():
            full = (model(fx).argmax(-1)
                    == fy).float().mean().item()
        if full == 1.0:
            return {"grok_step": grok_step, "exact": True}
    return {"grok_step": None, "exact": False}


def train_composite(table, seed=0, max_steps=20000):
    xs, ys = [], []
    for a1 in range(8):
        for b1 in range(8):
            for a2 in range(8):
                for b2 in range(8):
                    xs.append([a1, b1, a2, b2])
                    ys.append(table[table[a1][b1]][table[a2][b2]])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
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


def main():
    t0 = time.time()
    print("E43: THE BAND PRECISION SWEEP — the entropy law's edges")
    print("pre-registered seq 226\n")

    # the tables
    perm = [3, 0, 7, 4, 1, 6, 5, 2]  # a fixed non-identity permutation
    tables = {
        "T5": make_T(5), "T6": make_T(6), "T7": make_T(7),
        "T8": make_T(8), "PD": make_PD(perm),
    }
    print("--- table entropies (phoneme / composite labels) ---")
    for name, t in tables.items():
        pe = table_entropy(t)
        ce = composite_label_entropy(t)
        print(f"  {name}: table H={pe:.4f}  composite label H={ce:.4f}")
        log("table", task=name, table_entropy=pe,
            composite_label_entropy=ce)
    save()

    # P3: the phoneme floor
    print("\n=== P3: the phoneme floor (2 seeds each) ===")
    for name, t in tables.items():
        for seed in (0, 1):
            r = grok_phoneme(t, name, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            print(f"  {name} s{seed}: grok={g} exact={r['exact']}")
            log("phoneme", task=name, seed=seed,
                grok_step=r["grok_step"], exact=r["exact"])
        save()

    # P1 + P2: the composites, 4 seeds each
    print("\n=== P1/P2: self-composites (4 seeds each) ===")
    fam = {}
    for name, t in tables.items():
        grokked = 0
        steps_list = []
        for seed in range(4):
            r = train_composite(t, seed=seed)
            ok = r["grok_step"] is not None
            grokked += ok
            steps_list.append(r["grok_step"])
            g = r["grok_step"] if ok else "never"
            print(f"  {name}({name}) s{seed}: grok={g} "
                  f"test={r['final_test']}")
            log("composite", task=name, seed=seed, **r)
        fam[name] = grokked
        save()

    print("\n=== VERDICTS (pre-registered seq 226) ===")
    t5, t6, t7, t8, pd = (fam.get(k, 0) for k in
                          ("T5", "T6", "T7", "T8", "PD"))
    p1 = (t5 == 4 and t6 in (2, 3) and t7 in (1, 2) and t8 == 0)
    monotone = t5 >= t6 >= t7 >= t8
    p2 = pd == 0
    p3 = all(r["exact"] for r in RESULTS if r["kind"] == "phoneme")
    print(f"  family grok counts: {fam}")
    print(f"  P1 monotone onset (4/2-3/1-2/0): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"(monotone: {monotone})")
    print(f"  P2 permutation invariance (PD walls 0/4): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'}")
    print(f"  P3 phoneme floor (all exact): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'}")
    log("verdicts", P1_onset=p1, monotone=monotone, P2_invariance=p2,
        P3_floor=p3, fam=fam)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
