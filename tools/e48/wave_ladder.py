"""wave_ladder.py — E48: the wave ladder — frequency, amplitude,
phase as circuits.

The sovereign's question: circuits simulating sound waves. The
overnight laws split the three axes into three regimes:

  FREQ     the aligned case (T_k tables ARE frequency tables)
  AMP      the monotone case (the freest regime)
  PHASE    the carry (E39's Pauli phase wall)

Stimulus: 8-sample waveform x[n] = A*sin(2*pi*f*n/8 + phi),
encoded as an 8-token sequence, each sample quantized to 8 signed
levels (the ternary-substrate convention: {-3..4} as classes).

Arms (the E39/E43 protocol, d64/phi1/wd0.5/20k cap, 6 attempts,
full-domain-exact gate):
  A  FREQ-k    f in {1..k}, A=1, phi=0          k=4, 8
  B  AMP-k     A evenly spaced, f=1, phi=0       k=4, 8
  C  PHASE-ref phi = 2*pi*j/k vs fixed reference k=4, 8
  D  REL-PHASE two-component interference        k=8 (the wall cell)
  E  PD-FREQ   FREQ-8 with scrambled labels      (the control)

Pre-registration: tsh-512 seq 241.
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
NS = 8           # samples per stimulus
LEVELS = 8       # quantization classes


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


# ── the stimulus encoding ───────────────────────────────────────────

def quantize(x):
    """A sample in [-1, 1] -> one of LEVELS classes in {0..7}
    (signed clamp: -1 -> 0, +1 -> 7)."""
    c = int(round((x + 1.0) / 2.0 * (LEVELS - 1)))
    return max(0, min(LEVELS - 1, c))


def wave_tokens(A, f, phi):
    """The 8-sample waveform as an 8-token sequence."""
    return [quantize(A * math.sin(2 * math.pi * f * n / NS + phi))
            for n in range(NS)]


def rel_wave_tokens(phi):
    """Two-component interference at equal amplitude."""
    return [quantize(0.5 * math.sin(2 * math.pi * n / NS)
                     + 0.5 * math.sin(2 * math.pi * n / NS + phi))
            for n in range(NS)]


# ── the task builders ───────────────────────────────────────────────

def build_freq(k):
    xs = [wave_tokens(1.0, f, 0.0) for f in range(1, k + 1)]
    ys = list(range(k))
    return xs, ys


def build_amp(k):
    As = [0.15 + (1.0 - 0.15) * j / (k - 1) for j in range(k)]
    xs = [wave_tokens(A, 1, 0.0) for A in As]
    ys = list(range(k))
    return xs, ys


def build_phase(k):
    xs = [wave_tokens(1.0, 1, 2 * math.pi * j / k) for j in range(k)]
    ys = list(range(k))
    return xs, ys


def build_relphase(k):
    xs = [rel_wave_tokens(2 * math.pi * j / k) for j in range(k)]
    ys = list(range(k))
    return xs, ys


PD_PERM = [3, 0, 7, 4, 1, 6, 5, 2]  # E43's fixed scramble


def build_pd_freq(k):
    xs, ys = build_freq(k)
    ys = [PD_PERM[y] for y in ys]
    return xs, ys


# ── the trainer (E43's protocol; small domains -> full-batch,
#    leave-one-out test is meaningless at k<=8, so the gate is
#    FULL-DOMAIN exactness directly: the phoneme-gate convention) ────

def grok(xs, ys, name, seed=0, max_steps=20000):
    """Train to classify the k wave classes; the gate is exactness
    on the full (tiny) domain — every class learned. The model's
    positional embedding is 4-token (the timeline's convention), so
    each 8-sample waveform is presented as its two halves
    concatenated — [first 4 samples, last 4 samples] — the same
    pair-of-pairs interface every table task uses."""
    k = len(set(ys))
    # The model's positional embedding is 4-token (the timeline's
    # convention): each 8-sample waveform is presented as two
    # 4-token views (first half, second half) with logits summed.
    x1 = torch.tensor([x[:4] for x in xs], dtype=torch.long)
    x2 = torch.tensor([x[4:] for x in xs], dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    best_acc, best_step = 0.0, None
    for attempt in range(6):
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(LEVELS, k, d=64, lattice="phi1")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        for step in range(1, max_steps + 1):
            out = model(x1) + model(x2)   # two-view logits, summed
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                with torch.no_grad():
                    acc = ((model(x1) + model(x2)).argmax(-1)
                           == y).float().mean().item()
                if acc == 1.0:
                    return {"grok_step": step, "exact": True,
                            "attempt": seed + attempt}
        with torch.no_grad():
            acc = ((model(x1) + model(x2)).argmax(-1)
                   == y).float().mean().item()
        if acc > best_acc:
            best_acc, best_step = acc, step
    return {"grok_step": None, "exact": False,
            "best_acc": round(best_acc, 4)}


def main():
    t0 = time.time()
    print("E48: THE WAVE LADDER — frequency, amplitude, phase")
    print("pre-registered seq 241\n")

    # stimulus sanity: show the FREQ-4 and REL-PHASE tokens
    print("--- stimulus sanity (first cells) ---")
    print(f"  f=1: {wave_tokens(1.0, 1, 0.0)}")
    print(f"  f=2: {wave_tokens(1.0, 2, 0.0)}")
    print(f"  rel phi=0:   {rel_wave_tokens(0.0)}")
    print(f"  rel phi=pi:  {rel_wave_tokens(math.pi)}")
    log("stimulus", f1=wave_tokens(1.0, 1, 0.0),
        rel_pi=rel_wave_tokens(math.pi))
    save()

    arms = [
        ("FREQ-4", build_freq, 4),
        ("FREQ-8", build_freq, 8),
        ("AMP-4", build_amp, 4),
        ("AMP-8", build_amp, 8),
        ("PHASE-4", build_phase, 4),
        ("PHASE-8", build_phase, 8),
        ("REL-PHASE-8", build_relphase, 8),
        ("PD-FREQ-8", build_pd_freq, 8),
    ]

    print("\n--- the ladder (2 seeds each, exact gate) ---")
    results = {}
    for name, builder, k in arms:
        xs, ys = builder(k)
        for seed in (0, 1):
            r = grok(xs, ys, name, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            extra = "" if r["exact"] else \
                f" (best {r['best_acc']})"
            print(f"  {name} s{seed}: grok={g}{extra}")
            log("ladder", task=name, seed=seed,
                grok_step=r["grok_step"], exact=r["exact"],
                best_acc=r.get("best_acc"))
        results[name] = r
        save()

    # ── verdicts ──
    print("\n=== VERDICTS (pre-registered seq 241) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "ladder"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    freq4, freq8 = exact("FREQ-4"), exact("FREQ-8")
    amp4, amp8 = exact("AMP-4"), exact("AMP-8")
    ph4, ph8 = exact("PHASE-4"), exact("PHASE-8")
    rel8 = exact("REL-PHASE-8")
    pd8 = exact("PD-FREQ-8")
    p1 = (freq4[0] >= 1 and freq8[0] >= 1 and amp4[0] >= 1
          and ph4[0] >= 1)
    p2 = rel8[0] == 0
    p3 = pd8[0] == 0 and freq8[0] >= 1
    print(f"  P1 (axis split: FREQ+AMP grok, PHASE-ref groks at k=4): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"(FREQ {freq4}/{freq8}, AMP {amp4}/{amp8}, "
          f"PHASE {ph4}/{ph8})")
    print(f"  P2 (REL-PHASE walls — the interference carry): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} ({rel8[0]}/{rel8[1]})")
    print(f"  P3 (PD control walls where FREQ groks): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(PD {pd8[0]}/{pd8[1]}, FREQ-8 {freq8[0]}/{freq8[1]})")
    log("verdicts", P1_axis_split=p1, P2_relphase_walls=p2,
        P3_pd_control=p3,
        cells={n: exact(n) for n, _, _ in arms})
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
