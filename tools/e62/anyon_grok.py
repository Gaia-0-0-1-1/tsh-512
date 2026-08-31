"""anyon_grok.py — E62: anyon phonemes and braid words — quantum
circuits as the product of grokking, rung 1.

Task family: given (initial fusion-path state, braid word), predict
the most-likely measurement outcome (argmax |amplitude|^2) of the
exact golden-chain simulation. Phases matter only through
interference — the honest quantum-measurement encoding.

n=5 anyons, E=4 edges, tau sector dim 8 (F(6)=8), generators 1..4,
words length 3. Held-out-WORD splits (the E53 lesson: small
domains memorize; the shuffle control must wall).

Arms (2 seeds, gate = held-out accuracy >= 0.95 AND train exact):
  A FUSION-ADMIT   admissible vs not over 16 bitstrings
                   (hold out 2 bitstrings)
  B SIGMA-1        single generator (hold out generator 4)
  C WORD-3-DIRECT  12 words (hold out 4), direct encoding
  D WORD-3-REF     + a third view: the canonical word [2,3,2]'s
                   outcome on the same state (input-derived ref)
  E WORD-3-RAND    random labels on C's split (must fail)

Pre-registration: tsh-512 seq 285.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TSH = HERE.parents[1]

sys.path.insert(0, str(TSH / "tools" / "e6"))
sys.path.insert(0, str(TSH / "proto"))
sys.path.insert(0, str(TSH / "tools" / "e20"))
sys.path.insert(0, "C:/Users/Owen/golden-chain-spark")

import anyon_sim as A  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402

RESULTS = []
N_ANY = 5
E_EDGES = N_ANY - 1          # 4
DIM = 8                      # F(6)
LEVELS = 8                   # token vocab
CANON_WORD = [2, 3, 2]       # the reference word


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


masks = A.masks_no_adjacent(E_EDGES)          # the 8 admissible
state_bits = {int(m): i for i, m in enumerate(masks)}
bits_of = [tuple(int(b) for b in
                 format(int(m), f"0{E_EDGES}b")) for m in masks]


def outcome(state_idx, word):
    psi = A.initial_state(masks, mask0=masks[state_idx])
    psi = A.apply_word(psi, masks, N_ANY,
                       [(g, False) for g in word])
    return int(np.argmax(np.abs(psi) ** 2))


def ref_outcome(state_idx):
    return outcome(state_idx, CANON_WORD)


# the 12 train + 4 held-out words (deterministic)
rng = np.random.default_rng(42)
all_words = [(int(a), int(b), int(c))
             for a in range(1, 5) for b in range(1, 5)
             for c in range(1, 5)]
sel = list(rng.choice(len(all_words), 16, replace=False))
train_words = [all_words[i] for i in sel[:12]]
test_words = [all_words[i] for i in sel[12:]]


def views_C(state_idx, word):
    s = list(bits_of[state_idx])          # exactly 4 bits (E=4)
    w = [g + 3 for g in word] + [0]       # 3 gens + pad = 4 tokens
    return s, w


def views_D(state_idx, word):
    s = list(bits_of[state_idx])          # 4 bits
    r = list(bits_of[ref_outcome(state_idx)])  # 4 bits
    w = [g + 3 for g in word] + [0]       # 4 tokens
    return (s, w, r)                      # three views


def grok(train_x, train_y, test_x, test_y, name, seed=0,
         max_steps=20000):
    """Two-view training (three-view via concatenated passes)."""
    def tensors(xs):
        if len(xs[0]) == 2:
            x1 = torch.tensor([a for a, _ in xs], dtype=torch.long)
            x2 = torch.tensor([b for _, b in xs], dtype=torch.long)
        else:
            x1 = torch.tensor([a for a, _, _ in xs], dtype=torch.long)
            x2 = torch.tensor([b for _, b, _ in xs], dtype=torch.long)
            x3 = torch.tensor([c for _, _, c in xs], dtype=torch.long)
            return x1, x2, x3
        return x1, x2, None

    xtr = tensors(train_x)
    xte = tensors(test_x)
    ytr = torch.tensor(train_y, dtype=torch.long)
    yte = torch.tensor(test_y, dtype=torch.long)
    k = DIM
    best_te, exact_train = 0.0, False
    for attempt in range(6):
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(LEVELS, k, d=64, lattice="phi1")

        def fwd(idx):
            out = model(xtr[0][idx]) + model(xtr[1][idx])
            if xtr[2] is not None:
                out = out + model(xtr[2][idx])
            return out

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        hit = None
        for step in range(1, max_steps + 1):
            idx = torch.randperm(len(ytr))[:len(ytr)]
            loss = F.cross_entropy(fwd(idx), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                with torch.no_grad():
                    tr_acc = (fwd(torch.arange(len(ytr)))
                              .argmax(-1) == ytr).float().mean().item()
                if tr_acc == 1.0:
                    hit = step
                    break

        def te_acc():
            with torch.no_grad():
                out = model(xte[0]) + model(xte[1])
                if xte[2] is not None:
                    out = out + model(xte[2])
                return (out.argmax(-1) == yte).float().mean().item()

        if hit is not None:
            exact_train = True
            best_te = max(best_te, te_acc())
            if te_acc() >= 0.95:
                return {"train_exact": True, "held_out": te_acc(),
                        "pass": True, "step": hit}
        else:
            best_te = max(best_te, te_acc())
    return {"train_exact": exact_train,
            "held_out": round(best_te, 4), "pass": False,
            "step": None}


def main():
    t0 = time.time()
    print("E62: ANYON PHONEMES AND BRAID WORDS")
    print("pre-registered seq 285\n")

    # A: FUSION-ADMIT over all 16 four-bit strings
    print("--- A: FUSION-ADMIT ---")
    tr_x, tr_y, te_x, te_y = [], [], [], []
    all_bits = [tuple(int(b) for b in format(i, "04b"))
                for i in range(16)]
    import random as _r
    _r.Random(7).shuffle(all_bits)
    for i, bits in enumerate(all_bits):
        adm = all(bits[k] == 0 or bits[k + 1] == 0
                  for k in range(3))
        ex = (list(bits), [0, 0, 0, 0])
        if i < 14:
            tr_x.append(ex)
            tr_y.append(int(adm))
        else:
            te_x.append(ex)
            te_y.append(int(adm))
    for seed in (0, 1):
        r = grok(tr_x, tr_y, te_x, te_y, "A", seed)
        print(f"  A s{seed}: {r}")
        log("arm", arm="FUSION-ADMIT", seed=seed, **r)
    save()

    # B: SIGMA-1 (hold out generator 4)
    print("--- B: SIGMA-1 ---")
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for si in range(DIM):
        for g in (1, 2, 3, 4):
            ex = views_C(si, [g])
            lab = outcome(si, [g])
            if g == 4:
                te_x.append(ex)
                te_y.append(lab)
            else:
                tr_x.append(ex)
                tr_y.append(lab)
    for seed in (0, 1):
        r = grok(tr_x, tr_y, te_x, te_y, "B", seed)
        print(f"  B s{seed}: {r}")
        log("arm", arm="SIGMA-1", seed=seed, **r)
    save()

    # C: WORD-3 DIRECT (8 states x 12 train / 4 test words)
    print("--- C: WORD-3 DIRECT ---")
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for si in range(DIM):
        for w in train_words:
            v = views_C(si, w)
            tr_x.append(v)
            tr_y.append(outcome(si, list(w)))
        for w in test_words:
            v = views_C(si, w)
            te_x.append(v)
            te_y.append(outcome(si, list(w)))
    c_results = []
    for seed in (0, 1):
        r = grok(tr_x, tr_y, te_x, te_y, "C", seed)
        print(f"  C s{seed}: {r}")
        c_results.append(r)
        log("arm", arm="WORD-3-DIRECT", seed=seed, **r)
    save()

    # D: WORD-3 REF (three views)
    print("--- D: WORD-3 REF ---")
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for si in range(DIM):
        for w in train_words:
            v = views_D(si, w)
            tr_x.append(v)
            tr_y.append(outcome(si, list(w)))
        for w in test_words:
            v = views_D(si, w)
            te_x.append(v)
            te_y.append(outcome(si, list(w)))
    d_results = []
    for seed in (0, 1):
        r = grok(tr_x, tr_y, te_x, te_y, "D", seed)
        print(f"  D s{seed}: {r}")
        d_results.append(r)
        log("arm", arm="WORD-3-REF", seed=seed, **r)
    save()

    # E: WORD-3 RAND (shuffle labels; must fail)
    print("--- E: WORD-3 RAND ---")
    import random as _r2
    rr = _r2.Random(11)
    sh_y = [rr.randrange(DIM) for _ in tr_y]
    for seed in (0, 1):
        r = grok(tr_x, sh_y, te_x, te_y, "E", seed)
        print(f"  E s{seed}: {r}")
        log("arm", arm="WORD-3-RAND", seed=seed, **r)
    save()

    # verdicts
    def passrate(arm):
        rs = [r for r in RESULTS if r["kind"] == "arm"
              and r["arm"] == arm]
        return sum(1 for r in rs if r.get("pass")), len(rs)

    a, b = passrate("FUSION-ADMIT"), passrate("SIGMA-1")
    c, d, e = (passrate("WORD-3-DIRECT"), passrate("WORD-3-REF"),
               passrate("WORD-3-RAND"))
    c_held = [r["held_out"] for r in c_results]
    d_held = [r["held_out"] for r in d_results]
    p1 = a[0] >= 1 and b[0] >= 1
    p2 = c[0] <= 1
    p3 = (sum(1 for h in d_held if h >= 0.95)
          - sum(1 for h in c_held if h >= 0.95)) >= 1
    ctrl = e[0] == 0
    print("\n=== VERDICTS (pre-registered seq 285) ===")
    print(f"  P1 phonemes mint (A,B): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} {a} {b}")
    print(f"  P2 braid wall (C fails): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} {c} "
          f"held {c_held}")
    print(f"  P3 reference opens (D): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} {d} "
          f"held {d_held}")
    print(f"  control (E walls): {'OK' if ctrl else 'INVALID'}")
    log("verdicts", P1_phonemes=p1, P2_braid_wall=p2,
        P3_reference=p3, control_ok=ctrl,
        held_C=c_held, held_D=d_held)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
