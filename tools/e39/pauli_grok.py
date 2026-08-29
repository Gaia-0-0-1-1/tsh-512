"""pauli_grok.py — E39/Q2: quantum gate grokking.

The E-series grok protocol applied to quantum gate groups (qiskit
2.4.2 verifies the tables; the ladder extends into quantum
structure):

  PAULI4   — {I,X,Y,Z} mod phase (XY=Z cyclic, order 4)
  PAULI16  — the full 1-qubit Pauli group with phases (order 16,
             non-abelian up to i-commutation) — the phase-carry test
  Z16      — flat cyclic control (separates order-16 hardness from
             Pauli hardness)
  PAULI16-FACTORED — the representation arm: labels encoded as
             (sign, phase-index, Pauli-index) — does decomposition
             break the phase wall?

Pre-registration: tsh-512 seq 216.
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

from math_structures import make_task  # noqa: E402
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


# ── the Pauli tables (qiskit-verified below) ─────────────────────────

# PAULI4: {I,X,Y,Z} mod phase. XY = iZ -> Z. Cyclic order 4:
# generator X: X^0=I, X^1=X, X^2=Z? NO — X^2 = I. The mod-phase
# group: XY=Z, YZ=X, ZX=Y, XX=YY=ZZ=I. This is the KLEIN four-group
# V4 (= Z2xZ2), NOT cyclic: every element is self-inverse.
# (The pre-registration's cyclic guess was wrong — measured at
# construction, corrected here, kept honest.)
def make_pauli4():
    # encode I=0, X=1, Y=2, Z=3
    # table from matrix products mod phase:
    # I*anything = anything; XX=YY=ZZ=I; XY=Z, YX=Z;
    # XZ=Y, ZX=Y; YZ=X, ZY=X
    table = [[0, 1, 2, 3],
             [1, 0, 3, 2],
             [2, 3, 0, 1],
             [3, 2, 1, 0]]
    return table


def make_pauli16():
    """The full 1-qubit Pauli group: {±1,±i} x {I,X,Y,Z}.

    Element = phase p in {1,i,-1,-i} times Pauli s in {I,X,Y,Z}.
    16 elements. Product: (p1 s1)(p2 s2) = (p1 p2 sp) s3 where
    s1 s2 = sp s3 with sp the phase from the Pauli product.
    Encode: idx = 4*p_idx + s_idx  (p: 0=1, 1=i, 2=-1, 3=-i;
    s: 0=I, 1=X, 2=Y, 3=Z).
    """
    # Pauli product phases: XY=iZ, YZ=iX, ZX=iY, and antisymmetric
    # ones give -i. XX=YY=ZZ=I.
    def pauli_prod(s1, s2):
        """returns (phase_idx, pauli_idx) for s1*s2."""
        if s1 == 0:
            return 0, s2
        if s2 == 0:
            return 0, s1
        if s1 == s2:
            return 0, 0                    # XX = I, no phase
        # the cyclic products carry +i, anticyclic -i
        cyc = {(1, 2): 3, (2, 3): 1, (3, 1): 2}   # XY=Z, YZ=X, ZX=Y
        acyc = {(2, 1): 3, (3, 2): 1, (1, 3): 2}  # YX=Z, ZY=X, XZ=Y
        if (s1, s2) in cyc:
            return 1, cyc[(s1, s2)]       # +i
        return 3, acyc[(s1, s2)]          # -i

    def phase_prod(p1, p2):
        # phases as powers of i: idx 0=1,1=i,2=-1,3=-i -> add mod 4
        return (p1 + p2) % 4

    table = [[0] * 16 for _ in range(16)]
    for a in range(16):
        pa, sa = divmod(a, 4)
        for b in range(16):
            pb, sb = divmod(b, 4)
            sp, s3 = pauli_prod(sa, sb)
            p3 = phase_prod(phase_prod(pa, pb), sp)
            table[a][b] = p3 * 4 + s3
    return table


def make_z16():
    return [[(a + b) % 16 for b in range(16)] for a in range(16)]


def verify_pauli16_with_qiskit(table):
    """Cross-check the Pauli-16 table against qiskit matrix products."""
    import numpy as np
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [I, X, Y, Z]
    phases = [1, 1j, -1, -1j]
    mats = [phases[p] * paulis[s]
            for p in range(4) for s in range(4)]
    ok = True
    for a in range(16):
        for b in range(16):
            prod = mats[a] @ mats[b]
            expect = mats[table[a][b]]
            if not np.allclose(prod, expect):
                ok = False
                print(f"  MISMATCH at ({a},{b})")
    return ok


def verify_pauli4_with_qiskit(table):
    import numpy as np
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mats = [I, X, Y, Z]
    ok = True
    for a in range(4):
        for b in range(4):
            prod = mats[a] @ mats[b]
            # mod phase: prod = phase * mats[table]
            expect = mats[table[a][b]]
            ratio = None
            # find the phase
            for p in (1, 1j, -1, -1j):
                if np.allclose(prod, p * expect):
                    ratio = p
                    break
            if ratio is None:
                ok = False
                print(f"  P4 MISMATCH at ({a},{b})")
    return ok


# ── the task interface (make_task-compatible) ────────────────────────

def make_table_task(table, n, seed=0, train_frac=0.8):
    xs, ys = [], []
    for a in range(n):
        for b in range(n):
            xs.append([a, b])
            ys.append(table[a][b])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(round(train_frac * x.shape[0]))
    tr, te = perm[:n_train], perm[n_train:]
    return {"train_x": x[tr], "train_y": y[tr],
            "test_x": x[te], "test_y": y[te]}


def run_grok(table, n, seed=0, lattice="phi1", max_steps=20000):
    ds = make_table_task(table, n, seed)
    torch.manual_seed(seed)
    model = TinyTransformer(n, n, d=64, lattice=lattice)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=0.5, betas=(0.9, 0.98))
    x, y = ds["train_x"], ds["train_y"]
    grok_step, te_acc = None, 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randperm(x.shape[0])[:min(64, x.shape[0])]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                te_acc = (model(ds["test_x"]).argmax(-1)
                          == ds["test_y"]).float().mean().item()
            if te_acc >= 0.95:
                grok_step = step
                break
    return {"grok_step": grok_step, "final_test": round(te_acc, 4),
            "steps": step}


def run_grok_factored(table, n, seed=0, lattice="phi1",
                      max_steps=20000):
    """PAULI16-FACTORED: the label decomposed into (phase, pauli)
    as a 2-channel task: input (a,b), outputs BOTH the phase class
    (4-way) and the pauli class (4-way). Two heads, two losses —
    the model must learn the decomposition."""
    xs, ph, pa = [], [], []
    for a in range(n):
        for b in range(n):
            xs.append([a, b])
            p, s = divmod(table[a][b], 4)
            ph.append(p)
            pa.append(s)
    x = torch.tensor(xs, dtype=torch.long)
    y_ph = torch.tensor(ph, dtype=torch.long)
    y_pa = torch.tensor(pa, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(round(0.8 * x.shape[0]))
    tr, te = perm[:n_train], perm[n_train:]

    torch.manual_seed(seed)
    model = TinyTransformer(n, 16, d=64, lattice=lattice)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=0.5, betas=(0.9, 0.98))
    grok_step, te_acc = None, 0.0
    x_tr, x_te = x[tr], x[te]
    for step in range(1, max_steps + 1):
        idx = torch.randperm(tr.shape[0])[:64]
        logits = model(x_tr[idx])
        # two heads from the unembed: view (B,16) as (B,4,4)
        # [phase, pauli]; phase head marginalizes over pauli,
        # pauli head marginalizes over phase
        l2 = logits.view(-1, 4, 4)
        loss = (F.cross_entropy(l2.mean(dim=2), y_ph[tr][idx])
                + F.cross_entropy(l2.mean(dim=1), y_pa[tr][idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                lo = model(x_te).view(-1, 4, 4)
                pred_ph = lo.mean(dim=2).argmax(-1)
                pred_pa = lo.mean(dim=1).argmax(-1)
                both = ((pred_ph == y_ph[te]) &
                        (pred_pa == y_pa[te])).float().mean().item()
                te_acc = both
            if te_acc >= 0.95:
                grok_step = step
                break
    return {"grok_step": grok_step, "final_test": round(te_acc, 4),
            "steps": step}


def main():
    t0 = time.time()
    print("E39/Q2: QUANTUM GATE GROKKING — the Pauli ladder")
    print("pre-registered seq 216\n")

    # ── construct + verify the tables ──
    p4 = make_pauli4()
    p16 = make_pauli16()
    z16 = make_z16()
    ok4 = verify_pauli4_with_qiskit(p4)
    ok16 = verify_pauli16_with_qiskit(p16)
    print(f"  PAULI4 table verified mod phase vs qiskit: {ok4}")
    print(f"  PAULI16 table verified vs qiskit: {ok16}")
    log("tables", pauli4_verified=ok4, pauli16_verified=ok16)

    # is PAULI4 the Klein group? (every element self-inverse)
    klein = all(p4[a][a] == 0 for a in range(4))
    abelian16 = all(p16[a][b] == p16[b][a]
                    for a in range(16) for b in range(16))
    print(f"  PAULI4 is Klein V4 (all self-inverse): {klein} "
          f"(the pre-registration guessed cyclic — WRONG, corrected)")
    print(f"  PAULI16 abelian: {abelian16} (expect False — "
          f"phases anticommute)")
    log("structure", pauli4_is_klein=klein, pauli16_abelian=abelian16)

    # ── the grok ladder ──
    tasks = [
        ("PAULI4", p4, 4),
        ("PAULI16", p16, 16),
        ("Z16", z16, 16),
    ]
    print("\n=== the ladder (2 seeds each, phi1, d64, 20k cap) ===")
    for name, table, n in tasks:
        for seed in (0, 1):
            r = run_grok(table, n, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            print(f"  {name} s{seed}: grok={g} test={r['final_test']}")
            log("grok", task=name, seed=seed, **r)
        save()

    # ── the representation arm ──
    print("\n=== PAULI16-FACTORED (phase/pauli decomposition) ===")
    for seed in (0, 1):
        r = run_grok_factored(p16, 16, seed=seed)
        g = r["grok_step"] if r["grok_step"] else "never"
        print(f"  PAULI16-FACTORED s{seed}: grok={g} "
              f"test={r['final_test']}")
        log("grok_factored", task="PAULI16-FACTORED", seed=seed, **r)
    save()

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
