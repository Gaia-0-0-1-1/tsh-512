"""tone_composite.py — E50: tones — the first wave COMPOSITE.

A tone IS the composite frequency x phase. This experiment measures
the acoustic stack's first composition rung: does the product of
free wave axes follow the table laws?

Cells (the E48/E49 two-view protocol, exact gate, 2 seeds):
  TONE-4    f in {1,2} x phi in 4 steps   (1x2 bits joint)
  TONE-8    f in {1,2} x phi in 8 steps   (1x3 bits)
  TONE-16   f in {1..4} x phi in 4 steps  (2x2 bits)
  JOIN-16   the semantic join of minted FREQ and REF-PHASE circuits
            vs the 16-tone ground truth

Pre-registration: tsh-512 seq 246.
"""
import hashlib
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


def wave(A, f, phi):
    return [quantize(A * math.sin(2 * math.pi * f * n / NS + phi))
            for n in range(NS)]


def tone_views(f, phi):
    """The referenced pair: view1 = the zero-phase reference at
    frequency f (first 4 tokens), view2 = the phase-shifted wave
    (first 4 tokens). Their difference carries phi; view1's shape
    carries f."""
    ref = wave(1.0, f, 0.0)
    shifted = wave(1.0, f, phi)
    return ref[:4], shifted[:4]


def build_tone(freqs, n_phase):
    """The joint (f, phi) product task. Class index =
    f_idx * n_phase + phi_idx."""
    xs, ys = [], []
    for fi, f in enumerate(freqs):
        for pj in range(n_phase):
            phi = 2 * math.pi * pj / n_phase
            v1, v2 = tone_views(f, phi)
            xs.append((v1, v2))
            ys.append(fi * n_phase + pj)
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
                    return {"grok_step": step, "exact": True,
                            "model": model}
        with torch.no_grad():
            acc = ((model(x1) + model(x2)).argmax(-1)
                   == y).float().mean().item()
        best_acc = max(best_acc, acc)
    return {"grok_step": None, "exact": False,
            "best_acc": round(best_acc, 4), "model": None}


def grok_axis(builder, k, seed=0, max_steps=20000):
    """Grok one axis circuit (frequency or referenced phase)."""
    xs, ys = builder()
    assert len(set(ys)) == k
    x1 = torch.tensor([a for a, _ in xs], dtype=torch.long)
    x2 = torch.tensor([b for _, b in xs], dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
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
                    return model
    return None


def build_freq_axis():
    xs, ys = [], []
    for fi, f in enumerate(range(1, 5)):
        ref = wave(1.0, f, 0.0)
        xs.append((ref[:4], ref[4:]))
        ys.append(fi)
    return xs, ys


def build_phase_axis():
    xs, ys = [], []
    for pj in range(4):
        phi = 2 * math.pi * pj / 4
        v1, v2 = tone_views(1, phi)
        xs.append((v1, v2))
        ys.append(pj)
    return xs, ys


def join_tone(freq_model, phase_model, freqs, n_phase):
    """The semantic join: the frequency circuit reads view1 (the
    reference's shape -> f), the phase circuit reads (view1, view2)
    (the pair -> phi). The composite's output = f_idx * n_phase +
    phi_idx — the product structure assembled from the parts'
    predictions, zero training on the composite."""
    correct, total = 0, 0
    preds = []
    for fi, f in enumerate(freqs):
        for pj in range(n_phase):
            phi = 2 * math.pi * pj / n_phase
            v1, v2 = tone_views(f, phi)
            x1 = torch.tensor([v1], dtype=torch.long)
            x2 = torch.tensor([v2], dtype=torch.long)
            with torch.no_grad():
                # the frequency circuit: the reference's two halves
                ref_full = wave(1.0, f, 0.0)
                fx1 = torch.tensor([ref_full[:4]], dtype=torch.long)
                fx2 = torch.tensor([ref_full[4:]], dtype=torch.long)
                f_pred = (freq_model(fx1)
                          + freq_model(fx2)).argmax(-1).item()
                # the phase circuit: the referenced pair
                p_pred = (phase_model(x1)
                          + phase_model(x2)).argmax(-1).item()
            joint = f_pred * n_phase + p_pred
            expect = fi * n_phase + pj
            preds.append(joint)
            correct += joint == expect
            total += 1
    return correct / total, preds


def main():
    t0 = time.time()
    print("E50: TONES — the first wave COMPOSITE (frequency x "
          "referenced phase)")
    print("pre-registered seq 246\n")

    # ── the direct product tasks ──
    cells = [
        ("TONE-4", [1, 2], 4),      # 1 x 2 bits
        ("TONE-8", [1, 2], 8),      # 1 x 3 bits
        ("TONE-16", [1, 2, 3, 4], 4),  # 2 x 2 bits
    ]
    print("--- the direct product tasks ---")
    for name, freqs, n_phase in cells:
        xs, ys = build_tone(freqs, n_phase)
        for seed in (0, 1):
            r = grok(xs, ys, name, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            extra = "" if r["exact"] else f" (best {r['best_acc']})"
            print(f"  {name} s{seed}: grok={g}{extra}")
            log("tone", task=name, seed=seed,
                grok_step=r["grok_step"], exact=r["exact"],
                best_acc=r.get("best_acc"))
        save()

    # ── the join ──
    print("\n--- the semantic join of axis circuits ---")
    freq_model = grok_axis(build_freq_axis, 4)
    phase_model = grok_axis(build_phase_axis, 4)
    if freq_model is None or phase_model is None:
        print("  axis circuit failed to mint — join skipped")
        log("join", skipped=True,
            freq_minted=freq_model is not None,
            phase_minted=phase_model is not None)
    else:
        print("  both axis circuits minted exactly")
        acc, preds = join_tone(freq_model, phase_model,
                               [1, 2, 3, 4], 4)
        print(f"  JOIN-16 accuracy on the joint domain: {acc:.4f}")
        log("join", acc=round(acc, 4),
            fingerprint=hashlib.sha256(
                canon(preds).encode()).hexdigest()[:16])
    save()

    # ── verdicts ──
    print("\n=== VERDICTS (pre-registered seq 246) ===")
    def exact(name):
        rs = [r for r in RESULTS if r["kind"] == "tone"
              and r["task"] == name]
        return sum(1 for r in rs if r["exact"]), len(rs)

    t4, t8, t16 = exact("TONE-4"), exact("TONE-8"), exact("TONE-16")
    join_rec = [r for r in RESULTS if r["kind"] == "join"
                and not r.get("skipped")]
    join_acc = join_rec[0]["acc"] if join_rec else None
    p1 = t4[0] >= 1
    p2 = t16[0] <= 1
    p3 = join_acc is not None and join_acc == 1.0
    print(f"  P1 (TONE-4 groks directly): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} ({t4[0]}/{t4[1]})")
    print(f"  P2 (TONE-16 walls/coin-flips): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} ({t16[0]}/{t16[1]})")
    print(f"  P3 (the join computes the tone exactly): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(join acc={join_acc})")
    log("verdicts", P1_tone4=p1, P2_tone16_walls=p2,
        P3_join=p3, cells={"TONE-4": t4, "TONE-8": t8,
                           "TONE-16": t16}, join_acc=join_acc)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
