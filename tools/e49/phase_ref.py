"""phase_ref.py — E49: the reference-token phase encoding + the
correctly-sited PD control.

E48's falsifications demand two fixes:
  1. absolute phase walls at every size — is that an ENCODING
     failure (no visible anchor) or a capacity failure? Give each
     stimulus an explicit reference: view1 = the zero-phase wave,
     view2 = the phase-shifted wave. Their difference IS the phase.
  2. the PD control was mis-sited at k=8 (base task also walls).
     Site it at k=4 where FREQ-4 groks: PD-FREQ-4.

Arms (the E48 protocol, 2 seeds, exact gate):
  REF-PHASE-4 / REF-PHASE-8   reference+shifted as the two views
  PD-FREQ-4                   scrambled labels at the free cell
  PHASE-4                     the known wall (replication)

Pre-registration: tsh-512 seq 243.
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


def wave(A, f, phi):
    return [quantize(A * math.sin(2 * math.pi * f * n / NS + phi))
            for n in range(NS)]


# ── the stimulus builders ───────────────────────────────────────────

def build_ref_phase(k):
    """View1 = the zero-phase reference wave (f=1, A=1); view2 = the
    phase-shifted wave. The model sees both; their difference is
    the phase."""
    xs = []
    for j in range(k):
        ref = wave(1.0, 1, 0.0)
        shifted = wave(1.0, 1, 2 * math.pi * j / k)
        xs.append((ref, shifted))
    ys = list(range(k))
    return xs, ys


def build_phase(k):
    """E48's PHASE-ref: the shifted wave alone, two views of its own
    halves (the known wall)."""
    xs = []
    for j in range(k):
        w = wave(1.0, 1, 2 * math.pi * j / k)
        xs.append((w[:4], w[4:]))
    ys = list(range(k))
    return xs, ys


PD_PERM = [3, 0, 7, 4, 1, 6, 5, 2]


def build_pd_freq(k):
    """FREQ-k with scrambled output labels, sited where the base
    task groks (k=4). The scramble is a permutation OF range(k):
    same entropy, same vocab, renamed classes."""
    xs = []
    for f in range(1, k + 1):
        w = wave(1.0, f, 0.0)
        xs.append((w[:4], w[4:]))
    rng = random.Random(7)
    perm = list(range(k))
    rng.shuffle(perm)
    ys = [perm[j] for j in range(k)]
    return xs, ys


# ── the trainer (E48's two-view protocol) ───────────────────────────

def grok(xs, ys, name, seed=0, max_steps=20000):
    """The two views are each 4-token (the model's L=4 interface):
    view1 and view2 are each HALVES — for the reference encoding,
    view1 = the reference wave's first half, view2 = the shifted
    wave's first half (each 4 tokens); the model reads both."""
    k = len(set(ys))
    x1 = torch.tensor([a[:4] for a, _ in xs], dtype=torch.long)
    x2 = torch.tensor([b[:4] for _, b in xs], dtype=torch.long)
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
    print("E49: THE REFERENCE-TOKEN PHASE ENCODING + the sited PD"
          " control")
    print("pre-registered seq 243\n")

    arms = [
        ("REF-PHASE-4", build_ref_phase, 4),
        ("REF-PHASE-8", build_ref_phase, 8),
        ("PD-FREQ-4", build_pd_freq, 4),
        ("PHASE-4", build_phase, 4),      # the known wall, replication
    ]

    print("--- stimulus sanity ---")
    xs, ys = build_ref_phase(4)
    print(f"  ref (phi=0):      {xs[0][0]}")
    print(f"  shifted (phi=pi/2): {xs[1][1]}")
    xs2, ys2 = build_pd_freq(4)
    print(f"  PD-FREQ-4 labels: {ys2}")
    log("stimulus", ref=xs[0][0], shifted_pi_2=xs[1][1],
        pd_labels=ys2)
    save()

    print("\n--- the ladder (2 seeds each, exact gate) ---")
    for name, builder, k in arms:
        xs, ys = builder(k)
        for seed in (0, 1):
            r = grok(xs, ys, name, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            extra = "" if r["exact"] else f" (best {r['best_acc']})"
            print(f"  {name} s{seed}: grok={g}{extra}")
            log("ladder", task=name, seed=seed,
                grok_step=r["grok_step"], exact=r["exact"],
                best_acc=r.get("best_acc"))
        save()

    print("\n=== VERDICTS (pre-registered seq 243) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "ladder"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    ref4, ref8 = exact("REF-PHASE-4"), exact("REF-PHASE-8")
    pd4 = exact("PD-FREQ-4")
    ph4 = exact("PHASE-4")
    p1 = ref4[0] >= 1 and ph4[0] == 0
    p2 = pd4[0] == 0
    p3 = ref8[0] <= 1
    print(f"  P1 (reference opens the phase gate): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"(REF-PHASE-4 {ref4[0]}/{ref4[1]}, "
          f"PHASE-4 {ph4[0]}/{ph4[1]})")
    print(f"  P2 (PD control walls at the free site): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} "
          f"({pd4[0]}/{pd4[1]})")
    print(f"  P3 (3-bit boundary holds for phase): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(REF-PHASE-8 {ref8[0]}/{ref8[1]})")
    log("verdicts", P1_reference_opens=p1, P2_pd_walls=p2,
        P3_bit_boundary=p3,
        cells={"REF-PHASE-4": ref4, "REF-PHASE-8": ref8,
               "PD-FREQ-4": pd4, "PHASE-4": ph4})
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
