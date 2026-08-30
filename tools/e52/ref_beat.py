"""ref_beat.py — E52: the reference-transposition fix.

E51 measured the invariance wall: BEAT (classify |f1-f2| across
varied pairs) walls at 0.9375. Does the E49 pattern repeat — does
an explicit reference open this wall too?

Arms (the E48-E51 two-view protocol, exact gate, 2 seeds):
  REF-BEAT-4     view1 = a FIXED anchor pair (1,2); view2 = the
                 test pair. The beat becomes a comparison.
  CLASS-ANCHOR   view1 = an anchor whose beat EQUALS the test's
                 (the label visible; the floor test).
  BEAT-4         the control (must replicate 0.9375).
  SUM-4          the control (must replicate 200 steps).

Pre-registration: tsh-512 seq 250.
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
    return [quantize(0.5 * math.sin(2 * math.pi * f1 * n / NS)
                     + 0.5 * math.sin(2 * math.pi * f2 * n / NS))
            for n in range(NS)]


def beat_pairs():
    """E51's BEAT set: 4 pairs per beat class."""
    pairs = []
    for b in (0, 1, 2, 3):
        cls_pairs = []
        for f1 in range(1, 8):
            f2 = f1 + b
            if 1 <= f2 <= 7:
                cls_pairs.append((f1, f2))
        for f1, f2 in cls_pairs[:4]:
            pairs.append(((f1, f2), b))
    return pairs


# ── the arm builders ────────────────────────────────────────────────

def build_ref_beat():
    """View1 = the FIXED anchor (1,2)'s first half; view2 = the test
    pair's first half. The beat as a comparison against a constant
    reference."""
    anchor = two_component(1, 2)
    xs, ys = [], []
    for (f1, f2), b in beat_pairs():
        w = two_component(f1, f2)
        xs.append((anchor[:4], w[:4]))
        ys.append(b)
    return xs, ys


def build_class_anchor():
    """View1 = an anchor whose beat EQUALS the test pair's (the
    label is visible in the reference; the floor test). The anchor
    for class b is the FIRST pair of that class (deterministic),
    and the test pair is a DIFFERENT pair of the same class when
    available."""
    by_class = {}
    for (f1, f2), b in beat_pairs():
        by_class.setdefault(b, []).append((f1, f2))
    xs, ys = [], []
    for b, pairs in sorted(by_class.items()):
        anchor_pair = pairs[0]
        anchor = two_component(*anchor_pair)
        # test pairs: all pairs of the class (anchor included —
        # matching the anchor to itself is the trivial cell)
        for f1, f2 in pairs:
            w = two_component(f1, f2)
            xs.append((anchor[:4], w[:4]))
            ys.append(b)
    return xs, ys


def build_beat():
    """E51's original: two views of the test pair's own halves."""
    xs, ys = [], []
    for (f1, f2), b in beat_pairs():
        w = two_component(f1, f2)
        xs.append((w[:4], w[4:]))
        ys.append(b)
    return xs, ys


def build_sum():
    """E51's SUM control."""
    xs, ys = [], []
    for f1 in range(1, 7):
        for f2 in range(f1, 8):
            s = (f1 + f2) % 8
            if s == 0 or s > 4:
                continue
            w = two_component(f1, f2)
            xs.append((w[:4], w[4:]))
            ys.append(s - 1)
    by_class = {}
    for x, y in zip(xs, ys):
        by_class.setdefault(y, []).append(x)
    xs2, ys2 = [], []
    for y, exs in sorted(by_class.items()):
        for x in exs[:6]:
            xs2.append(x)
            ys2.append(y)
    return xs2, ys2


# ── the trainer ─────────────────────────────────────────────────────

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
    print("E52: THE REFERENCE-TRANSPOSITION FIX — does the E49 "
          "pattern repeat one level up?")
    print("pre-registered seq 250\n")

    arms = [
        ("REF-BEAT-4", build_ref_beat),
        ("CLASS-ANCHOR-4", build_class_anchor),
        ("BEAT-4", build_beat),        # the control (the wall)
        ("SUM-4", build_sum),          # the control (the free cell)
    ]

    # sanity
    from collections import Counter
    print("--- stimulus sanity ---")
    for name, builder in arms:
        xs, ys = builder()
        print(f"  {name}: {len(xs)} exemplars, "
              f"classes {sorted(Counter(ys).items())}")
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

    print("\n=== VERDICTS (pre-registered seq 250) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "ladder"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    ref, cls_a = exact("REF-BEAT-4"), exact("CLASS-ANCHOR-4")
    beat, summ = exact("BEAT-4"), exact("SUM-4")
    p1 = ref[0] >= 1 and beat[0] == 0
    p2 = beat[0] == 0 and summ[0] == 2
    p3 = cls_a[0] == 2
    print(f"  P1 (reference opens the invariance gate): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"(REF-BEAT {ref[0]}/{ref[1]}, BEAT {beat[0]}/{beat[1]})")
    print(f"  P2 (controls replicate): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} "
          f"(BEAT {beat[0]}/{beat[1]}, SUM {summ[0]}/{summ[1]})")
    print(f"  P3 (the class-anchor floor groks): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"({cls_a[0]}/{cls_a[1]})")
    log("verdicts", P1_reference_opens=p1, P2_controls=p2,
        P3_floor=p3,
        cells={"REF-BEAT": ref, "CLASS-ANCHOR": cls_a,
               "BEAT": beat, "SUM": summ})
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
