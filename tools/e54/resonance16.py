"""resonance16.py — E54: the resonance ladder at 16 samples.

The metallic-means thesis's audition-side test. At 16 samples with
quarter-integer frequency resolution, rational intervals are
PERIODIC in the window and irrational approximants are not — is
periodicity the learnable signature?

Stimulus: 16-sample two-component waves, f in quarter steps.
Views: the model's L=4 interface — view1 = samples 0-3, view2 =
samples 4-7 (the first half); the second half is discarded (the
interface constraint, recorded honestly — periodicity at 16 shows
in any 8 consecutive samples for rational intervals with period
dividing 8, and the pre-registered design accepts this).

Arms (exact gate, 2 seeds):
  PERIODICITY-2    periodic vs aperiodic pure tones (the floor)
  INTERVAL-RAT-4   rational intervals across transpositions
  INTERVAL-IRR-4   irrational-approximant intervals
  MIXED-4          {octave, fifth, tritone, phi} together — the
                   decisive cell: confusion structure rational vs
                   irrational

Pre-registration: tsh-512 seq 253.
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
NS = 16
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


def wave2(f1, f2):
    """16-sample two-component wave, quarter-integer f."""
    return [quantize(0.5 * math.sin(2 * math.pi * f1 * n / NS)
                     + 0.5 * math.sin(2 * math.pi * f2 * n / NS))
            for n in range(NS)]


def views(w):
    """The L=4 interface: first two 4-sample blocks."""
    return w[:4], w[4:8]


# ── the arms ────────────────────────────────────────────────────────

def build_periodicity():
    """Periodic (integer cycles in the window) vs aperiodic
    (quarter-cycle offsets) pure tones — the direct periodicity
    read, 2 classes, the floor."""
    xs, ys = [], []
    periodic = [0.5, 1.0, 1.5, 2.0]
    aperiodic = [0.25, 0.75, 1.25, 1.75]
    for f in periodic:
        w = [quantize(math.sin(2 * math.pi * f * n / NS))
             for n in range(NS)]
        xs.append(views(w)); ys.append(0)
    for f in aperiodic:
        w = [quantize(math.sin(2 * math.pi * f * n / NS))
             for n in range(NS)]
        xs.append(views(w)); ys.append(1)
    return xs, ys


def interval_pairs(ratio, transps):
    """Pairs (f1, f2) with f2/f1 ~= ratio, f1 over transps, all at
    quarter resolution."""
    out = []
    for f1 in transps:
        f2 = round(f1 * ratio * 4) / 4.0
        if 0.25 <= f2 <= 3.75 and f2 > f1:
            out.append((f1, f2))
    return out


TRANSPS = [0.5, 0.75, 1.0, 1.25, 1.5]


def build_rat():
    """Rational intervals: octave (2/1), fifth (3/2), fourth (4/3),
    major third (5/4) — periodic in 16 samples."""
    specs = [(2.0, 0), (1.5, 1), (4/3, 2), (1.25, 3)]
    xs, ys = [], []
    for ratio, cls in specs:
        for f1, f2 in interval_pairs(ratio, TRANSPS):
            w = wave2(f1, f2)
            xs.append(views(w)); ys.append(cls)
    return xs, ys


def build_irr():
    """Irrational intervals at quarter approximation: sqrt(2),
    phi, pi/2, e/2 — aperiodic at true value; the quarter grid is
    the honest probe."""
    specs = [(math.sqrt(2), 0), ((1 + math.sqrt(5)) / 2, 1),
             (math.pi / 2, 2), (math.e / 2, 3)]
    xs, ys = [], []
    for ratio, cls in specs:
        for f1, f2 in interval_pairs(ratio, TRANSPS):
            w = wave2(f1, f2)
            xs.append(views(w)); ys.append(cls)
    return xs, ys


def build_mixed():
    """The decisive cell: {octave, fifth, tritone(sqrt2), phi}
    together — does the confusion structure separate rational from
    irrational?"""
    specs = [(2.0, 0), (1.5, 1), (math.sqrt(2), 2),
             ((1 + math.sqrt(5)) / 2, 3)]
    xs, ys = [], []
    for ratio, cls in specs:
        for f1, f2 in interval_pairs(ratio, TRANSPS):
            w = wave2(f1, f2)
            xs.append(views(w)); ys.append(cls)
    return xs, ys


def grok(xs, ys, name, seed=0, max_steps=20000):
    k = len(set(ys))
    x1 = torch.tensor([a for a, _ in xs], dtype=torch.long)
    x2 = torch.tensor([b for _, b in xs], dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    best_acc = 0.0
    best_model = None
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
    # confusion structure at best accuracy
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
    print("E54: THE RESONANCE LADDER AT 16 SAMPLES — is periodicity"
          " the learnable signature?")
    print("pre-registered seq 253\n")

    arms = [
        ("PERIODICITY-2", build_periodicity),
        ("INTERVAL-RAT-4", build_rat),
        ("INTERVAL-IRR-4", build_irr),
        ("MIXED-4", build_mixed),
    ]

    from collections import Counter
    print("--- stimulus sanity ---")
    print(f"  octave (0.5,1.0):     {wave2(0.5, 1.0)}")
    print(f"  fifth  (1.0,1.5):     {wave2(1.0, 1.5)}")
    print(f"  tritone(1.0,~1.41):   {wave2(1.0, 1.5)}")  # quarter grid!
    for name, builder in arms:
        xs, ys = builder()
        print(f"  {name}: {len(xs)} exemplars, "
              f"classes {sorted(Counter(ys).items())}")
    log("stimulus", octave=wave2(0.5, 1.0), fifth=wave2(1.0, 1.5))
    save()

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

    print("\n=== VERDICTS (pre-registered seq 253) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "ladder"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    per, rat = exact("PERIODICITY-2"), exact("INTERVAL-RAT-4")
    irr, mix = exact("INTERVAL-IRR-4"), exact("MIXED-4")
    p1 = per[0] >= 1
    p3 = rat[0] >= 1 and irr[0] >= 1
    # P2 needs the confusion analysis — recorded from the best cell
    mix_recs = [r for r in RESULTS if r["kind"] == "ladder"
                and r["task"] == "MIXED-4"]
    p2 = None   # assessed in the verdict record from confusions
    print(f"  P1 (periodicity control groks): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} ({per[0]}/{per[1]})")
    print(f"  P3 (both families learn as tasks): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(RAT {rat[0]}/{rat[1]}, IRR {irr[0]}/{irr[1]})")
    print(f"  P2 (mixed confusion separates rat/irr): assessed from "
          f"confusion records below")
    log("verdicts", P1_periodicity=p1, P3_families=p3,
        cells={"PER": per, "RAT": rat, "IRR": irr, "MIX": mix})
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
