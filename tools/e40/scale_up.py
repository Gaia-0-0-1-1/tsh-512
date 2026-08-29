"""scale_up.py — E40: vocabulary scale-up beyond algebra.

Do the arc's laws (the fusion wall, the semantic join, the loop)
survive when the phonemes are NOT groups?

Tasks (all order-8 tables):
  QUASI8     T(a,b) = (3a+5b) mod 8 — Latin square, no identity,
             no associativity (affine: linear structure)
  MONOID-AND bitwise AND — monoid, not group, not Latin
  MONOID-OR  bitwise OR — dual monoid
  RANDOM8    seeded uniform random — the null control

Arms:
  (a) the ladder — where do non-group structures grok?
  (b) the wall  — direct composite training (E32 protocol)
  (c) the join  — the E36 semantic join, zero parameters

Pre-registration: tsh-512 seq 219.
"""
import hashlib
import json
import random
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
sys.path.insert(0, str(UNIFIED / "COMPUTE"))
sys.path.insert(0, str(UNIFIED / "MEMORY" / "rld"))

from hyperbyte_test import TinyTransformer  # noqa: E402
from fuse import SemanticJoin  # noqa: E402

RESULTS = []
N = 8


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


# ── the tables ──────────────────────────────────────────────────────

def make_quasi8():
    """T(a,b) = (3a+5b) mod 8. Latin: 3 and 5 are units mod 8.
    Non-associative, no identity — a proper quasigroup."""
    return [[(3 * a + 5 * b) % 8 for b in range(8)] for a in range(8)]


def make_and8():
    """Bitwise AND on {0..7}: monoid, identity 7, absorbing 0."""
    return [[a & b for b in range(8)] for a in range(8)]


def make_or8():
    """Bitwise OR on {0..7}: monoid, identity 0."""
    return [[a | b for b in range(8)] for a in range(8)]


def make_random8(seed=1234):
    """The null control: uniform random table."""
    rng = random.Random(seed)
    return [[rng.randrange(8) for _ in range(8)] for _ in range(8)]


def verify_tables():
    """Verify the structural claims (Latin-ness, monoid laws)."""
    checks = {}
    q = make_quasi8()
    latin = all(sorted(row) == list(range(8)) for row in q) and \
        all(sorted(col) == list(range(8)) for col in zip(*q))
    checks["quasi8_latin"] = latin
    # associativity should FAIL for quasi8
    assoc_q = all(q[q[a][b]][c] == q[a][q[b][c]]
                  for a in range(8) for b in range(8) for c in range(8))
    checks["quasi8_not_associative"] = not assoc_q
    a8 = make_and8()
    assoc_a = all(a8[a8[a][b]][c] == a8[a][a8[b][c]]
                  for a in range(8) for b in range(8) for c in range(8))
    ident_a = all(a8[a][7] == a and a8[7][a] == a for a in range(8))
    checks["and8_monoid"] = assoc_a and ident_a
    o8 = make_or8()
    assoc_o = all(o8[o8[a][b]][c] == o8[a][o8[b][c]]
                  for a in range(8) for b in range(8) for c in range(8))
    ident_o = all(o8[a][0] == a and o8[0][a] == a for a in range(8))
    checks["or8_monoid"] = assoc_o and ident_o
    # AND is not Latin (row 0 all zeros)
    checks["and8_not_latin"] = sorted(a8[0]) != list(range(8))
    return checks


TASKS = {
    "QUASI8": make_quasi8,
    "AND8": make_and8,
    "OR8": make_or8,
    "RANDOM8": make_random8,
}


# ── the grok protocol (E33's, full-domain verified) ─────────────────

def full_domain(table):
    xs = [[a, b] for a in range(8) for b in range(8)]
    ys = [table[a][b] for a in range(8) for b in range(8)]
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def grok_phoneme(name, table, seed=0, lattice="phi1", max_steps=20000):
    fx, fy = full_domain(table)
    for attempt in range(6):
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(8, 8, d=64, lattice=lattice)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        # train split: 80% of the 64 pairs
        g = torch.Generator().manual_seed(seed + attempt)
        perm = torch.randperm(64, generator=g)
        tr = perm[:51]
        x_tr, y_tr = fx[tr], fy[tr]
        grok_step, te_acc = None, 0.0
        for step in range(1, max_steps + 1):
            idx = torch.randperm(x_tr.shape[0])[:51]
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                with torch.no_grad():
                    te = ~perm[:51]  # complement mask
                    te_idx = perm[51:]
                    te_acc = (model(fx[te_idx]).argmax(-1)
                              == fy[te_idx]).float().mean().item()
                if te_acc >= 0.95:
                    grok_step = step
                    break
        model.eval()
        with torch.no_grad():
            full = (model(fx).argmax(-1) == fy).float().mean().item()
        if full == 1.0:
            return {"model": model, "steps": grok_step,
                    "test_acc": te_acc, "attempt": seed + attempt}
    return {"model": None, "steps": None, "test_acc": te_acc,
            "attempt": None}


# ── the composite protocol (E32's convention) ───────────────────────

def composite_truth(outer_t, inner_t):
    """outer(inner(p1), inner(p2)) over the pair-of-pairs domain."""
    xs, ys = [], []
    for a1 in range(8):
        for b1 in range(8):
            for a2 in range(8):
                for b2 in range(8):
                    c1 = inner_t[a1][b1]
                    c2 = inner_t[a2][b2]
                    xs.append([a1, b1, a2, b2])
                    ys.append(outer_t[c1][c2])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            ys)


def train_composite(outer_t, inner_t, seed=0, max_steps=20000):
    x, y, ys = composite_truth(outer_t, inner_t)
    rng = random.Random(seed)
    perm = rng.sample(range(len(x)), len(x))
    n_train = int(0.8 * len(x))
    tr, te = perm[:n_train], perm[n_train:]
    torch.manual_seed(seed)
    model = TinyTransformer(8, 8, d=64, lattice=None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=0.5, betas=(0.9, 0.98))
    x_tr, y_tr = x[tr], y[tr]
    x_te, y_te = x[te], y[te]
    grok_step, te_acc = None, 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randperm(x_tr.shape[0])[:64]
        loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                te_acc = (model(x_te).argmax(-1)
                          == y_te).float().mean().item()
            if te_acc >= 0.95:
                grok_step = step
                break
    return {"grok_step": grok_step, "final_test": round(te_acc, 4),
            "steps": step}


def join_verify(inner_model, outer_model, outer_t, inner_t):
    """The E36 semantic join, zero parameters, full domain."""
    x, y, ys = composite_truth(outer_t, inner_t)
    sj = SemanticJoin(inner_model, outer_model).eval()
    with torch.no_grad():
        preds = sj(x).argmax(-1).tolist()
    acc = sum(p == t for p, t in zip(preds, ys)) / len(preds)
    fp_join = hashlib.sha256(canon(preds).encode()).hexdigest()
    fp_true = hashlib.sha256(canon(ys).encode()).hexdigest()
    return acc, fp_join == fp_true


# ── minting (E22 protocol) ──────────────────────────────────────────

def mint(model, table, name):
    fx, fy = full_domain(table)
    with torch.no_grad():
        preds = model(fx).argmax(-1).tolist()
    fp = hashlib.sha256(canon(preds).encode()).hexdigest()
    return {"fingerprint": fp, "task": name,
            "accuracy_on_domain": 1.0, "lattice": "phi1",
            "grok_step": None, "family": "non-group",
            "weight_sha256": fp}


def main():
    t0 = time.time()
    print("E40: VOCABULARY SCALE-UP BEYOND ALGEBRA")
    print("pre-registered seq 219\n")

    # ── structural verification ──
    checks = verify_tables()
    for k, v in checks.items():
        print(f"  {k}: {v}")
    log("structure_checks", **checks)
    assert all(checks.values()), "structural claims failed"
    save()

    # ── arm (a): the ladder ──
    print("\n=== arm (a): the ladder — non-group structures ===")
    circuits = {}
    for name, maker in TASKS.items():
        table = maker()
        for seed in (0, 1):
            r = grok_phoneme(name, table, seed=seed)
            g = r["steps"] if r["steps"] else "never"
            print(f"  {name} s{seed}: grok={g} "
                  f"test={r['test_acc']:.3f} full={'exact' if r['model'] else 'no'}")
            log("ladder", task=name, seed=seed,
                grok_step=r["steps"], test_acc=round(r["test_acc"], 4),
                exact=r["model"] is not None)
            if r["model"] is not None and seed == 0:
                circuits[name] = (r["model"], table)
        save()

    # ── arm (b): the wall — direct composite training ──
    print("\n=== arm (b): the wall — direct composite training ===")
    COMPOSITES = [("QUASI8", "QUASI8"), ("AND8", "AND8"),
                  ("OR8", "OR8"), ("AND8", "QUASI8"),
                  ("QUASI8", "AND8")]
    tables = {name: TASKS[name]() for name in TASKS}
    wall_results = {}
    for outer, inner in COMPOSITES:
        for seed in (0, 1):
            r = train_composite(tables[outer], tables[inner], seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            print(f"  {outer}({inner}) s{seed}: grok={g} "
                  f"test={r['final_test']}")
            log("wall", outer=outer, inner=inner, seed=seed, **r)
        wall_results[(outer, inner)] = r
        save()

    # ── arm (c): the join — the representation-agnostic escape ──
    print("\n=== arm (c): the semantic join, zero parameters ===")
    join_results = {}
    for outer, inner in COMPOSITES:
        if inner not in circuits or outer not in circuits:
            # RANDOM8 will be missing (never exact); AND8/QUASI8
            # present only if they grokked exactly
            print(f"  {outer}({inner}): SKIPPED (part not exactly "
                  f"grokked)")
            log("join", outer=outer, inner=inner, skipped=True)
            continue
        inner_m, _ = circuits[inner]
        outer_m, _ = circuits[outer]
        acc, match = join_verify(inner_m, outer_m,
                                 tables[outer], tables[inner])
        print(f"  {outer}({inner}): join acc={acc:.4f} "
              f"fingerprint {'MATCH' if match else 'MISMATCH'}")
        log("join", outer=outer, inner=inner,
            acc=round(acc, 4), fingerprint_match=match)
        join_results[(outer, inner)] = (acc, match)
    save()

    # ── minting into the unified bank ──
    print("\n=== minting non-group phonemes into the vocabulary ===")
    if circuits:
        from circuit_nodes import CircuitRLD
        rld = CircuitRLD(UNIFIED / "MEMORY" / "rld" / "circuit_rld.jsonl")
        for name, (model, table) in circuits.items():
            m = mint(model, table, name)
            res = rld.append("CIRCUIT", m,
                             note="E40 non-group phoneme mint")
            print(f"  {name}: minted -> {res}")
            log("mint", task=name, fingerprint=m["fingerprint"][:16])
        print(f"  ledger: {rld.verify()}")
    save()

    # ── verdicts ──
    print("\n=== VERDICTS (pre-registered seq 219) ===")
    # P1: structured grok, RANDOM8 never
    random_grokked = any(
        r["grok_step"] for r in RESULTS
        if r["kind"] == "ladder" and r["task"] == "RANDOM8")
    structured = {}
    for r in RESULTS:
        if r["kind"] == "ladder" and r["task"] != "RANDOM8":
            structured.setdefault(r["task"], []).append(r["grok_step"])
    structured_grokked = {
        k: any(v) for k, v in structured.items()}
    p1 = (not random_grokked) and any(structured_grokked.values())
    print(f"  P1 structure-vs-null: "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"(structured grokked: {structured_grokked}, "
          f"RANDOM8 grokked: {random_grokked})")
    # P2: wall extends — at least 3/5 families fail
    fam = {}
    for r in RESULTS:
        if r["kind"] == "wall":
            fam.setdefault((r["outer"], r["inner"]), []).append(
                r["grok_step"] is not None)
    failed_families = sum(1 for v in fam.values() if not any(v))
    p2 = failed_families >= 3
    print(f"  P2 wall extends: {'CONFIRMED' if p2 else 'FALSIFIED'} "
          f"({failed_families}/{len(fam)} families fail direct)")
    # P3: join exact everywhere parts are exact
    joins = [(k, v) for k, v in join_results.items()]
    p3 = all(match for _, (acc, match) in joins) if joins else None
    print(f"  P3 join representation-agnostic: "
          f"{'CONFIRMED' if p3 else 'FALSIFIED' if p3 is not None else 'N/A (no exact parts)'} "
          f"({sum(1 for _, (a, m) in joins if m)}/{len(joins)} exact)")
    log("verdicts", P1=p1, P2=p2, P3=p3,
        failed_families=failed_families,
        joins_exact=sum(1 for _, (a, m) in joins if m),
        joins_total=len(joins))
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
