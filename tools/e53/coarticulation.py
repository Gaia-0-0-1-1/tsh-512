"""coarticulation.py — E53: phoneme-level coarticulation — the
reference law's third rung.

The same TARGET tone embedded in varying CONTEXTS — classify the
target invariantly. In phonology this is coarticulation: the same
phoneme changes shape with its neighbors.

Stimulus: 3-component waveform
  x[n] = 0.4*sin(2*pi*fc*n/8) + 0.4*sin(2*pi*ft*n/8)
       + 0.4*sin(2*pi*fo*n/8)
  (context fc, target ft in {1,2}, distractor fo)

Arms (the two-view protocol, exact gate, 2 seeds):
  COART-2-NOREF   the standard halves encoding (no reference)
  COART-2-REF     view1 = the CANONICAL-context embedding,
                  view2 = the ACTUAL-context embedding
  COART-2-FLOOR   no context variation (the floor control)
  COART-2-RAND    random labels (the ceiling-check control)

Pre-registration: tsh-512 seq 252.
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


def mix(fc, ft, fo):
    """The 3-component waveform, amplitude 0.4 each (sums within
    [-1.2, 1.2] -> clamped by quantize)."""
    return [quantize(0.4 * math.sin(2 * math.pi * fc * n / NS)
                     + 0.4 * math.sin(2 * math.pi * ft * n / NS)
                     + 0.4 * math.sin(2 * math.pi * fo * n / NS))
            for n in range(NS)]


# the context family: (fc, fo) pairs, target ft in {1, 2}
CONTEXTS = [(3, 4), (4, 5), (5, 6), (6, 7), (3, 6), (4, 7)]
CANONICAL = (3, 4)     # the fixed reference context


def build_noref():
    """The standard encoding: the mixture's own halves. The target
    must be read through context variation with NO reference."""
    xs, ys = [], []
    for ft in (1, 2):
        for fc, fo in CONTEXTS:
            w = mix(fc, ft, fo)
            xs.append((w[:4], w[4:]))
            ys.append(ft - 1)
    return xs, ys


def build_ref():
    """The reference encoding: view1 = the SAME target in the
    CANONICAL context, view2 = the actual-context embedding. The
    target's identity is invariant between the views; the context
    varies."""
    xs, ys = [], []
    for ft in (1, 2):
        for fc, fo in CONTEXTS:
            canonical = mix(*CANONICAL, ft)
            actual = mix(fc, ft, fo)
            xs.append((canonical[:4], actual[:4]))
            ys.append(ft - 1)
    return xs, ys


def build_floor():
    """No context variation: the target always in the canonical
    context — pure distinctness (the floor)."""
    xs, ys = [], []
    for ft in (1, 2):
        w = mix(*CANONICAL, ft)
        xs.append((w[:4], w[4:]))
        ys.append(ft - 1)
    return xs, ys


def build_rand():
    """Random labels on the no-ref stimulus (the ceiling check)."""
    xs, ys = build_noref()
    rng = random.Random(11)
    ys = [rng.randrange(2) for _ in ys]
    return xs, ys


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
    print("E53: COARTICULATION — the reference law's third rung")
    print("pre-registered seq 252\n")

    arms = [
        ("COART-2-NOREF", build_noref),
        ("COART-2-REF", build_ref),
        ("COART-2-FLOOR", build_floor),
        ("COART-2-RAND", build_rand),
    ]

    from collections import Counter
    print("--- stimulus sanity ---")
    print(f"  target f=1 in ctx (3,4): {mix(3, 1, 4)}")
    print(f"  target f=1 in ctx (6,7): {mix(6, 1, 7)}")
    print(f"  target f=2 in ctx (3,4): {mix(3, 2, 4)}")
    for name, builder in arms:
        xs, ys = builder()
        print(f"  {name}: {len(xs)} exemplars, "
              f"classes {sorted(Counter(ys).items())}")
    log("stimulus", f1_ctx34=mix(3, 1, 4), f1_ctx67=mix(6, 1, 7),
        f2_ctx34=mix(3, 2, 4))
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

    print("\n=== VERDICTS (pre-registered seq 252) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "ladder"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    noref, ref = exact("COART-2-NOREF"), exact("COART-2-REF")
    floor, rand = exact("COART-2-FLOOR"), exact("COART-2-RAND")
    p1 = noref[0] <= 1
    p2 = ref[0] >= 1 and noref[0] == 0
    p3 = floor[0] == 2 and rand[0] == 0
    print(f"  P1 (the coarticulation wall): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"({noref[0]}/{noref[1]})")
    print(f"  P2 (the reference opens the third rung): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} "
          f"(REF {ref[0]}/{ref[1]} vs NOREF {noref[0]}/{noref[1]})")
    print(f"  P3 (the controls): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(floor {floor[0]}/{floor[1]}, rand {rand[0]}/{rand[1]})")
    log("verdicts", P1_coart_wall=p1, P2_reference_opens=p2,
        P3_controls=p3,
        cells={"NOREF": noref, "REF": ref, "FLOOR": floor,
               "RAND": rand})
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
