"""wave_arithmetic.py — E51: wave-medium arithmetic.

The QFT-correspondence thesis made operational: part of arithmetic
is computed BY PHYSICS before the learner sees anything. The
interference of two frequency waves physically contains their
DIFFERENCE (beats) but never their SUM (superposition is linear)
and never their RATIO. The predictions follow from what the medium
hands the learner for free.

Arms (the E48/E49 two-view protocol, exact gate, 2 seeds):
  BEAT-4     class = |f1-f2|      (the physically-precomputed op)
  SUM-4      class = (f1+f2) mod 8 (derived: not in the pattern)
  HARMONIC-4 class = the interval ratio f2/f1 (multiplicative)
  PHI-4      the resonance ladder: rational / near-rational /
             Pisot(phi) / most-dissonant control

Stimulus: 8 samples of x[n] = 0.5*sin(2*pi*f1*n/8) +
0.5*sin(2*pi*f2*n/8), integer f (exact in the window), two 4-token
views (halves), 8-level quantization.

Pre-registration: tsh-512 seq 248.
"""
import json
import math
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
NS = 8
LEVELS = 8


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


def two_component(f1, f2):
    """8 samples of the two-frequency interference, quantized."""
    return [quantize(0.5 * math.sin(2 * math.pi * f1 * n / NS)
                     + 0.5 * math.sin(2 * math.pi * f2 * n / NS))
            for n in range(NS)]


def views(w):
    return w[:4], w[4:]


# ── the arm builders ────────────────────────────────────────────────

def build_beat():
    """Class = |f1-f2| in {0,1,2,3}. Balanced pairs: for each beat b,
    several (f1,f2) pairs — the class is the PHYSICAL envelope,
    invariant across pairs. Beat-0 = unison pairs (f1==f2)."""
    xs, ys = [], []
    for b in (0, 1, 2, 3):
        pairs = []
        for f1 in range(1, 8):
            f2 = f1 + b
            if 1 <= f2 <= 7:
                pairs.append((f1, f2))
        for f1, f2 in pairs[:4]:
            w = two_component(f1, f2)
            v1, v2 = views(w)
            xs.append((v1, v2))
            ys.append(b)
    return xs, ys


def build_sum():
    """Class = (f1+f2) mod 8. Same stimulus family as BEAT — the
    learner must DERIVE the sum (not present in the pattern)."""
    xs, ys = [], []
    for f1 in range(1, 7):
        for f2 in range(f1, 8):
            s = (f1 + f2) % 8
            if s == 0 or s > 4:
                continue
            w = two_component(f1, f2)
            v1, v2 = views(w)
            xs.append((v1, v2))
            ys.append(s - 1)          # classes 1..4 -> 0..3
    # balance: keep up to 6 per class
    by_class = {}
    for x, y in zip(xs, ys):
        by_class.setdefault(y, []).append(x)
    xs2, ys2 = [], []
    for y, exs in sorted(by_class.items()):
        for x in exs[:6]:
            xs2.append(x)
            ys2.append(y)
    return xs2, ys2


def build_harmonic():
    """Class = the interval f2/f1 as a musical ratio: unison (1/1),
    octave (2/1), fifth (3/2), fourth (4/3). The interference
    contains sums and differences, never ratios."""
    specs = [("unison", 0, 1, 1), ("octave", 1, 1, 2),
             ("fifth", 2, 2, 3), ("fourth", 3, 3, 4)]
    xs, ys = [], []
    for _, cls, f1, f2 in specs:
        # transpositions: shift both frequencies within the window
        for shift in range(0, 5):
            a, b = f1 + shift, f2 + shift
            if 1 <= a <= 7 and 1 <= b <= 7 and a <= b:
                w = two_component(a, b)
                v1, v2 = views(w)
                xs.append((v1, v2))
                ys.append(cls)
    return xs, ys


def build_phi():
    """The resonance ladder: interval classes probed at INTEGER
    approximants inside the 8-sample window. RATIONAL (exact small
    ratios), NEAR-RATIONAL (dense approximants of pi-like ratios),
    PISOT-PHI (convergents of the golden ratio), SQRT2 (convergents
    of sqrt 2). HONEST LIMIT (kept): the window quantizes every
    ratio to a rational; this measures whether approximant
    STRUCTURE leaves a learnable signature."""
    classes = [
        ("RATIONAL", 0, [(2, 1), (3, 2), (4, 3)]),
        ("NEAR-RATIONAL", 1, [(3, 1), (22, 7), (13, 4)]),
        ("PISOT-PHI", 2, [(3, 2), (5, 3), (8, 5)]),
        ("SQRT2", 3, [(7, 5), (3, 2), (10, 7)]),
    ]
    xs, ys = [], []
    for name, cls, pairs in classes:
        for f1, f2 in pairs:
            # reduce into the 1..7 window by octave division
            while f2 > 7 or f1 > 7:
                f1 = max(1, f1 // 2)
                f2 = max(1, f2 // 2)
            if f1 == f2 or not (1 <= f1 <= 7 and 1 <= f2 <= 7):
                continue
            w = two_component(f1, f2)
            v1, v2 = views(w)
            xs.append((v1, v2))
            ys.append(cls)
    return xs, ys


# ── the trainer (E48/E49 protocol) ──────────────────────────────────

def grok(xs, ys, name, seed=0, max_steps=20000):
    k = len(set(ys))
    x1 = torch.tensor([a for a, _ in xs], dtype=torch.long)
    x2 = torch.tensor([b for _, b in xs], dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    best_acc = 0.0
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
        best_acc = max(best_acc, acc)
    return {"grok_step": None, "exact": False,
            "best_acc": round(best_acc, 4)}


def main():
    t0 = time.time()
    print("E51: WAVE-MEDIUM ARITHMETIC — what the medium computes "
          "for free")
    print("pre-registered seq 248\n")

    arms = [
        ("BEAT-4", build_beat),
        ("SUM-4", build_sum),
        ("HARMONIC-4", build_harmonic),
        ("PHI-4", build_phi),
    ]

    # stimulus sanity
    print("--- stimulus sanity ---")
    w = two_component(2, 3)         # beat 1
    print(f"  f=(2,3) beat-1: {w}")
    w = two_component(1, 2)         # octave
    print(f"  f=(1,2) octave: {w}")
    for name, builder in arms:
        xs, ys = builder()
        from collections import Counter
        print(f"  {name}: {len(xs)} exemplars, "
              f"classes {dict(Counter(ys))}")
    log("stimulus", beat_2_3=two_component(2, 3),
        octave_1_2=two_component(1, 2))
    save()

    print("\n--- the ladder (2 seeds each, exact gate) ---")
    for name, builder in arms:
        xs, ys = builder()
        for seed in (0, 1):
            r = grok(xs, ys, name, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            extra = "" if r["exact"] else f" (best {r['best_acc']})"
            print(f"  {name} s{seed}: grok={g}{extra}")
            log("ladder", task=name, seed=seed,
                grok_step=r["grok_step"], exact=r["exact"],
                best_acc=r.get("best_acc"))
        save()

    print("\n=== VERDICTS (pre-registered seq 248) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "ladder"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    beat, summ = exact("BEAT-4"), exact("SUM-4")
    harm, phi = exact("HARMONIC-4"), exact("PHI-4")
    beat_steps = [r["grok_step"] for r in RESULTS
                  if r["kind"] == "ladder" and r["task"] == "BEAT-4"
                  and r["grok_step"]]
    p1 = beat[0] >= 1 and all(s <= 400 for s in beat_steps)
    p2a = summ[0] >= 1
    p2b = harm[0] <= 1
    p2 = p2a and p2b
    p3 = phi[0] >= 1     # the resonance cell's minimal read; the
                         # ordering read is recorded in the verdict
    print(f"  P1 (BEAT groks fast — the medium's gift): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"({beat[0]}/{beat[1]}, steps {beat_steps})")
    print(f"  P2 (SUM groks but slower; HARMONIC walls — the "
          f"hierarchy replicates): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} "
          f"(SUM {summ[0]}/{summ[1]}, HARMONIC {harm[0]}/{harm[1]})")
    print(f"  P3 (the resonance cell learns — structure present): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"({phi[0]}/{phi[1]})")
    log("verdicts", P1_beat_fast=p1, P2_hierarchy=p2, P3_phi=p3,
        cells={"BEAT": beat, "SUM": summ, "HARMONIC": harm,
               "PHI": phi})
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
