"""discrim_ladder.py — E54b: the fine-grained discrimination ladder.

E54's resonance question in runnable form (after the grid-collision
failure, seq 255): f1 fixed at 1.0, f2 across the eighth grid —
which frequency separations can the wave interface resolve, and
does the confusion structure follow interference symmetry?

PRE-FLIGHT FINDING (recorded before running): on the eighth grid
at 16 samples, EVERY f2 is periodic (all p/8 rationals) — the
periodic/aperiodic axis degenerates. The live structural axis is
SYMMETRY: which pairs produce patterns that are circular shifts or
reflections of each other (aliasing), vs genuinely distinct. The
experiment measures the discrimination curve and reads the
confusion structure against symmetry.

Arms (exact gate, 2 seeds):
  DISCRIM-8        all 8 f2 classes (the distinctness floor)
  DISCRIM-8-SHUF   scrambled labels (the E53-lesson control)
  ADJ-1500-1625    binary: 1.5 vs 1.625
  ADJ-1250-1375    binary: 1.25 vs 1.375
  ADJ-1875-2000    binary: 1.875 vs 2.0 (the period-16 vs 8 pair)

Pre-registration: tsh-512 seq 256.
"""
import json
import math
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

from hyperbyte_test import TinyTransformer  # noqa: E402

RESULTS = []
NS = 16
LEVELS = 8
F2S = [1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0]


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


def quantize(x):
    c = int(round((x + 1.0) / 2.0 * (LEVELS - 1)))
    return max(0, min(LEVELS - 1, c))


def wave2(f2):
    """f1 = 1.0 fixed; the 16-sample interference, two views of the
    first 8 samples."""
    w = [quantize(0.5 * math.sin(2 * math.pi * n / NS)
                  + 0.5 * math.sin(2 * math.pi * f2 * n / NS))
         for n in range(NS)]
    return w[:4], w[4:8]


def build_discrim():
    xs = [wave2(f) for f in F2S]
    ys = list(range(8))
    return xs, ys


def build_discrim_shuffle():
    xs, ys = build_discrim()
    rng = random.Random(11)
    ys = [rng.randrange(8) for _ in ys]
    # rebalance: ensure all 8 classes appear (8 exemplars, random
    # labels may drop classes) — use a permutation instead
    ys = list(range(8))
    rng.shuffle(ys)
    return xs, ys


def build_adjacent(fa, fb):
    """Binary discrimination of two f2 values, with a few
    transpositions of the WHOLE pair (f1 shifted too) for exemplar
    count."""
    xs, ys = [], []
    for f1 in (0.75, 1.0, 1.25):
        for f2, cls in ((fa, 0), (fb, 1)):
            w = [quantize(0.5 * math.sin(2 * math.pi * f1 * n / NS)
                          + 0.5 * math.sin(2 * math.pi * f2 * n / NS))
                 for n in range(NS)]
            xs.append((w[:4], w[4:8]))
            ys.append(cls)
    return xs, ys


def grok(xs, ys, name, seed=0, max_steps=20000):
    k = len(set(ys))
    x1 = torch.tensor([a for a, _ in xs], dtype=torch.long)
    x2 = torch.tensor([b for _, b in xs], dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    best_acc, best_model = 0.0, None
    for attempt in range(6):
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(LEVELS, k, d=64, lattice="phi1")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        for step in range(1, max_steps + 1):
            out = model(x1) + model(x2)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                with torch.no_grad():
                    acc = ((model(x1) + model(x2)).argmax(-1)
                           == y).float().mean().item()
                if acc == 1.0:
                    return {"grok_step": step, "exact": True}
        with torch.no_grad():
            acc = ((model(x1) + model(x2)).argmax(-1)
                   == y).float().mean().item()
        if acc > best_acc:
            best_acc, best_model = acc, model
    conf = None
    if best_model is not None:
        with torch.no_grad():
            preds = (best_model(x1) + best_model(x2)).argmax(-1)
        conf = {}
        for p, t in zip(preds.tolist(), y.tolist()):
            conf[f"{t}->{p}"] = conf.get(f"{t}->{p}", 0) + 1
    return {"grok_step": None, "exact": False,
            "best_acc": round(best_acc, 4), "confusion": conf}


def main():
    t0 = time.time()
    print("E54b: THE DISCRIMINATION LADDER — which f2 separations "
          "resolve?")
    print("pre-registered seq 256\n")

    # pre-flight symmetry scan: circular-shift aliasing between classes
    print("--- pre-flight: circular-shift aliasing ---")
    def raw(f2):
        return [round(0.5 * math.sin(2 * math.pi * n / NS)
                      + 0.5 * math.sin(2 * math.pi * f2 * n / NS), 3)
                for n in range(NS)]
    def shift(a, k):
        return a[-k:] + a[:-k] if k else a[:]
    raws = {f: raw(f) for f in F2S}
    alias = {}
    for f in F2S:
        partners = [g for g in F2S if g != f
                    and any(shift(raws[f], k) == raws[g]
                            for k in range(16))]
        if partners:
            alias[f] = partners
    print(f"  shift-aliased pairs: {alias or 'none'}")
    log("preflight_aliasing", aliasing={str(k): v
                                        for k, v in alias.items()})
    save()

    arms = [
        ("DISCRIM-8", build_discrim),
        ("DISCRIM-8-SHUF", build_discrim_shuffle),
        ("ADJ-1500-1625", lambda: build_adjacent(1.5, 1.625)),
        ("ADJ-1250-1375", lambda: build_adjacent(1.25, 1.375)),
        ("ADJ-1875-2000", lambda: build_adjacent(1.875, 2.0)),
    ]

    print("\n--- the ladder (2 seeds each, exact gate) ---")
    for name, builder in arms:
        xs, ys = builder()
        for seed in (0, 1):
            r = grok(xs, ys, name, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            extra = "" if r["exact"] else \
                f" (best {r['best_acc']} conf {r.get('confusion')})"
            print(f"  {name} s{seed}: grok={g}{extra}")
            log("ladder", task=name, seed=seed,
                grok_step=r["grok_step"], exact=r["exact"],
                best_acc=r.get("best_acc"),
                confusion=r.get("confusion"))
        save()

    print("\n=== VERDICTS (pre-registered seq 256) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "ladder"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    d8, shuf = exact("DISCRIM-8"), exact("DISCRIM-8-SHUF")
    a1, a2, a3 = (exact("ADJ-1500-1625"), exact("ADJ-1250-1375"),
                  exact("ADJ-1875-2000"))
    p1 = d8[0] >= 1
    p2 = shuf[0] <= 1
    adj = {"1.5v1.625": a1, "1.25v1.375": a2, "1.875v2.0": a3}
    p3 = None   # assessed from the ordering of the adjacent cells
    print(f"  P1 (distinctness floor, DISCRIM-8 groks): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} ({d8[0]}/{d8[1]})")
    print(f"  P2 (shuffle control fails): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} ({shuf[0]}/{shuf[1]})")
    print(f"  P3 (adjacent-pair ordering): {adj} — assessed in verdict")
    log("verdicts", P1_floor=p1, P2_shuffle=p2,
        cells={"DISCRIM8": d8, "SHUF": shuf, **adj})
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
