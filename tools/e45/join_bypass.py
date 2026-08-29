"""join_bypass.py — E45: does the semantic join bypass the
mismatch factor?

E44's three-factor law binds direct composite LEARNING. The H4
semantic join (zero-parameter construction) was verified on
Z4xZ2/Z2x2x2 — but never on the cross-vocab grid or scrambled
parts. This experiment tests the join on every E44-walled pair
whose parts grok exactly.

Arms:
  1. grok the phonemes (E33 protocol, full-domain exact required)
  2. the join arm: SemanticJoin on every walled pair with exact
     parts — cross-structural, non-adjacent cyclic, scrambled —
     verified against Cayley ground truth (full domain,
     fingerprint match)

Pre-registration: tsh-512 seq 232.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TSH = HERE.parents[1]
UNIFIED = TSH.parent / "unified"

sys.path.insert(0, str(TSH / "tools" / "e6"))
sys.path.insert(0, str(TSH / "proto"))
sys.path.insert(0, str(TSH / "tools" / "e20"))
sys.path.insert(0, str(TSH / "tools" / "e36"))
sys.path.insert(0, str(TSH / "tools" / "e38"))
sys.path.insert(0, str(UNIFIED / "COMPUTE"))

from math_structures import STRUCTURES  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402
from fuse import SemanticJoin  # noqa: E402
from join_of_joins import vocab_map  # noqa: E402
from h4_fusion import grok_circuit  # noqa: E402

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


def make_T(k):
    return [[(a + b) % k for b in range(8)] for a in range(8)]


PERM = [3, 0, 7, 4, 1, 6, 5, 2]


def make_PD():
    z8 = make_T(8)
    return [[PERM[v] for v in row] for row in z8]


# ── a general semantic join with the vocab_map interface ────────────

def join_logits(ma, mb, x, T=1.0):
    """The semantic join with E38's vocab_map interface: softmax
    over A's logits -> map into B's vocab -> B's embedding rows ->
    B's stack. Handles any vocab relationship."""
    na = ma.embed.num_embeddings
    nb = mb.embed.num_embeddings
    p1 = torch.softmax(ma(x[:, :2]) / T, dim=-1) @ vocab_map(na, nb)
    p2 = torch.softmax(ma(x[:, 2:]) / T, dim=-1) @ vocab_map(na, nb)
    w = mb.embed.weight
    e = torch.stack([p1 @ w, p2 @ w], dim=1)
    # B's stack from the embedding (h4_fusion.b_stack)
    from h4_fusion import b_stack
    return b_stack(mb, e)


def composite_truth(outer_t, inner_t):
    xs, ys = [], []
    for a1 in range(8):
        for b1 in range(8):
            for a2 in range(8):
                for b2 in range(8):
                    xs.append([a1, b1, a2, b2])
                    ys.append(outer_t[inner_t[a1][b1]][inner_t[a2][b2]])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            ys)


def verify_join(ma, mb, outer_t, inner_t):
    x, y, ys_true = composite_truth(outer_t, inner_t)
    with torch.no_grad():
        preds = join_logits(ma, mb, x).argmax(-1).tolist()
    acc = sum(p == t for p, t in zip(preds, ys_true)) / len(preds)
    fp_join = hashlib.sha256(canon(preds).encode()).hexdigest()
    fp_true = hashlib.sha256(canon(ys_true).encode()).hexdigest()
    return acc, fp_join == fp_true


def main():
    t0 = time.time()
    print("E45: DOES THE SEMANTIC JOIN BYPASS THE MISMATCH FACTOR?")
    print("pre-registered seq 232\n")

    # ── arm 1: the phonemes ──
    # grok_circuit works on STRUCTURES names; the T_k tables need
    # the generic path (E43's grok_phoneme). Use E43's protocol.
    sys.path.insert(0, str(TSH / "tools" / "e43"))
    from band_sweep import grok_phoneme  # noqa: E402

    tables = {"T5": make_T(5), "T6": make_T(6), "T7": make_T(7),
              "T8": make_T(8), "PD": make_PD(),
              "Z4xZ2": STRUCTURES["Z4xZ2"]["make"](),
              "Z2x2x2": STRUCTURES["Z2x2x2"]["make"]()}
    circuits = {}
    print("--- the phoneme gate (exact required) ---")
    for name, table in tables.items():
        got = None
        for seed in (0, 1):
            r = grok_phoneme(table, name, seed=seed)
            if r["exact"]:
                # re-grok to hold the model (grok_phoneme returns
                # metrics only) — inline it here
                got = _grok_and_hold(table, seed)
                break
        if got is not None:
            circuits[name] = got
            print(f"  {name}: exact (held)")
            log("phoneme", task=name, exact=True)
        else:
            print(f"  {name}: NOT EXACT — cells using it skipped")
            log("phoneme", task=name, exact=False)
    save()

    # ── arm 2: the join on E44's walled pairs ──
    PAIRS = [
        ("Z4xZ2", "Z2x2x2"), ("Z2x2x2", "Z4xZ2"),   # E32/E44 cells
        ("T5", "T7"), ("T5", "T8"), ("T6", "T8"),   # non-adjacent
        ("T7", "T8"), ("T8", "T7"), ("T8", "T6"),
        ("T7", "T6"),
        ("T8", "PD"), ("PD", "T8"),                 # scrambled
    ]
    print("\n--- the join arm: E44's walled pairs ---")
    exact_joins, tested = 0, 0
    for outer, inner in PAIRS:
        if outer not in circuits or inner not in circuits:
            print(f"  {outer}({inner}): SKIPPED (part not exact)")
            log("join", outer=outer, inner=inner, skipped=True)
            continue
        acc, match = verify_join(circuits[inner], circuits[outer],
                                 tables[outer], tables[inner])
        tested += 1
        exact_joins += match
        print(f"  {outer}({inner}): join acc={acc:.4f} "
              f"fingerprint {'MATCH' if match else 'MISMATCH'}")
        log("join", outer=outer, inner=inner,
            acc=round(acc, 4), fingerprint_match=match)
    save()

    # ── verdicts ──
    print("\n=== VERDICTS (pre-registered seq 232) ===")
    p1 = tested > 0 and exact_joins == tested
    print(f"  P1 (join bypasses mismatch, all exact): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"({exact_joins}/{tested} exact)")
    scrambled = [r for r in RESULTS if r["kind"] == "join"
                 and "PD" in (r.get("outer"), r.get("inner"))
                 and not r.get("skipped")]
    p2 = all(r["fingerprint_match"] for r in scrambled) \
        if scrambled else None
    print(f"  P2 (scrambled parts join exactly): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED' if p2 is not None else 'N/A'}")
    log("verdicts", P1_bypass=p1, P2_scrambled=p2,
        exact_joins=exact_joins, tested=tested)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


def _grok_and_hold(table, seed=0):
    """Grok and return the model (E43 protocol, full-domain exact)."""
    xs = [[a, b] for a in range(8) for b in range(8)]
    ys = [table[a][b] for a in range(8) for b in range(8)]
    fx = torch.tensor(xs, dtype=torch.long)
    fy = torch.tensor(ys, dtype=torch.long)
    for attempt in range(6):
        g = torch.Generator().manual_seed(seed + attempt)
        perm = torch.randperm(64, generator=g)
        tr, te = perm[:51], perm[51:]
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(8, 8, d=64, lattice="phi1")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        x_tr, y_tr = fx[tr], fy[tr]
        for step in range(1, 20001):
            idx = torch.randperm(x_tr.shape[0])[:51]
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                with torch.no_grad():
                    te_acc = (model(fx[te]).argmax(-1)
                              == fy[te]).float().mean().item()
                if te_acc >= 0.95:
                    break
        model.eval()
        with torch.no_grad():
            full = (model(fx).argmax(-1)
                    == fy).float().mean().item()
        if full == 1.0:
            return model
    return None


if __name__ == "__main__":
    main()
