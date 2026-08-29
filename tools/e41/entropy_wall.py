"""entropy_wall.py — E41: is the wall's predictor the inner's
output entropy?

Part 1 (retrospective, zero training): entropy features for all 21
measured composites (E32's 16 + E40's 5), correlated with grok
outcome.

Part 2 (prospective, the decisive cells): three new phonemes that
decouple entropy from monoid-ness —
  MUL8  multiplication mod 8 (monoid, MEDIUM entropy)
  MIN8  min semilattice (LOW entropy)
  MOD6  (a+b) mod 6 (monoid, NON-Latin, HIGH entropy, no
        absorbing element) — the decisive cell: if MOD6(MOD6)
        walls, entropy is the predictor; if it groks, monoid-
        ness/collapse is.

Pre-registration: tsh-512 seq 222.
"""
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TSH = HERE.parents[1]

sys.path.insert(0, str(TSH / "tools" / "e6"))
sys.path.insert(0, str(TSH / "proto"))
sys.path.insert(0, str(TSH / "tools" / "e20"))
sys.path.insert(0, str(TSH / "tools" / "e40"))

from math_structures import STRUCTURES  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402
from scale_up import (make_quasi8, make_and8, make_or8,  # noqa: E402
                      grok_phoneme, train_composite, composite_truth)

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


# ── the new tables ───────────────────────────────────────────────────

def make_mul8():
    """Multiplication mod 8: associative monoid, identity 1,
    zero absorbing. MEDIUM output entropy."""
    return [[(a * b) % 8 for b in range(8)] for a in range(8)]


def make_min8():
    """min(a,b): semilattice — associative, commutative, idempotent.
    LOW output entropy (skewed low)."""
    return [[min(a, b) for b in range(8)] for a in range(8)]


def make_mod6():
    """T(a,b) = (a+b) mod 6: associative monoid on {0..7}, identity
    0, NON-Latin (surjective onto 6 values, two inputs collide).
    HIGH output entropy (~2.58 bits), NO absorbing element —
    collapse without absorption. The decisive cell."""
    return [[(a + b) % 6 for b in range(8)] for a in range(8)]


# ── entropy features ────────────────────────────────────────────────

def entropy_bits(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log2(p)
    return round(h, 4)


def table_output_entropy(table):
    """H of the table's output distribution over its 64 cells."""
    return entropy_bits(Counter(v for row in table for v in row))


def composite_features(outer_t, inner_t, n_o, n_i):
    """The entropy features of a composite, E32's convention:
    domain n^4 with n = min(n_o, n_i); inner outputs reduced mod
    n_o before the outer sees them."""
    n = min(n_o, n_i)
    # what the outer actually sees: inner(a,b) % n_o over the
    # inner's cells that appear in the domain (a,b < n)
    inner_dist = Counter()
    for a in range(n):
        for b in range(n):
            inner_dist[inner_t[a][b] % n_o] += 1
    # composite labels over the full n^4 domain
    label_dist = Counter()
    for a1 in range(n):
        for b1 in range(n):
            for a2 in range(n):
                for b2 in range(n):
                    c1 = inner_t[a1][b1] % n_o
                    c2 = inner_t[a2][b2] % n_o
                    label_dist[outer_t[c1][c2]] += 1
    return {
        "inner_entropy": entropy_bits(inner_dist),
        "label_entropy": entropy_bits(label_dist),
        "domain_n": n,
        "vocab": n_o,
    }


# ── part 1: the retrospective correlation ───────────────────────────

def retrospective():
    print("=== part 1: retrospective — entropy vs grok, 21 families ===\n")
    # gather grok outcomes
    outcomes = {}  # (outer, inner) -> seeds_grokked
    with open(TSH / "tools" / "e32" / "results.jsonl",
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "composite":
                key = (r["outer"], r["inner"])
                outcomes[key] = outcomes.get(key, 0) + \
                    (1 if r.get("grok_step") else 0)
    with open(TSH / "tools" / "e40" / "results.jsonl",
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "wall":
                key = (r["outer"], r["inner"])
                outcomes[key] = outcomes.get(key, 0) + \
                    (1 if r.get("grok_step") else 0)

    tables = {name: STRUCTURES[name]["make"]() for name in STRUCTURES}
    tables.update({"QUASI8": make_quasi8(), "AND8": make_and8(),
                   "OR8": make_or8()})
    sizes = {name: STRUCTURES[name]["n"] for name in STRUCTURES}
    sizes.update({"QUASI8": 8, "AND8": 8, "OR8": 8})

    rows = []
    for (outer, inner), grokked in sorted(outcomes.items()):
        feats = composite_features(tables[outer], tables[inner],
                                    sizes[outer], sizes[inner])
        rows.append({"outer": outer, "inner": inner,
                     "grokked_seeds": grokked, **feats})
        print(f"  {outer}({inner}): grok={grokked}/2  "
              f"inner_H={feats['inner_entropy']:.3f}  "
              f"label_H={feats['label_entropy']:.3f}  "
              f"n={feats['domain_n']}")
        log("retro", outer=outer, inner=inner,
            grokked_seeds=grokked, **feats)

    # P1: separation test
    ever = [r for r in rows if r["grokked_seeds"] > 0]
    never = [r for r in rows if r["grokked_seeds"] == 0]
    mean = lambda rs, k: (round(sum(r[k] for r in rs) / len(rs), 4)
                          if rs else None)
    sep_inner = (mean(ever, "inner_entropy"),
                 mean(never, "inner_entropy"))
    sep_label = (mean(ever, "label_entropy"),
                 mean(never, "label_entropy"))
    print(f"\n  families with >=1 grok: {len(ever)}, never: {len(never)}")
    print(f"  mean inner entropy: grokked={sep_inner[0]} "
          f"never={sep_inner[1]}")
    print(f"  mean label entropy: grokked={sep_label[0]} "
          f"never={sep_label[1]}")
    p1 = (sep_inner[0] is not None and sep_inner[1] is not None
          and sep_inner[0] < sep_inner[1])
    print(f"  P1 separation (inner entropy): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'}")
    log("retro_verdict", P1_separation=p1,
        mean_inner_grokked=sep_inner[0],
        mean_inner_never=sep_inner[1],
        mean_label_grokked=sep_label[0],
        mean_label_never=sep_label[1])
    return rows


# ── part 2: the prospective decisive cells ──────────────────────────

def prospective():
    print("\n=== part 2: the decisive cells — entropy vs monoid-ness ===")
    new_tables = {"MUL8": make_mul8(), "MIN8": make_min8(),
                  "MOD6": make_mod6()}
    # structural checks
    mul = new_tables["MUL8"]
    assoc = all(mul[mul[a][b]][c] == mul[a][mul[b][c]]
                for a in range(8) for b in range(8) for c in range(8))
    ident = all(mul[a][1] == a and mul[1][a] == a for a in range(8))
    log("structure", MUL8_monoid=assoc and ident)
    mn = new_tables["MIN8"]
    assoc_mn = all(mn[mn[a][b]][c] == mn[a][mn[b][c]]
                   for a in range(8) for b in range(8) for c in range(8))
    log("structure", MIN8_semilattice=assoc_mn)
    md = new_tables["MOD6"]
    assoc_md = all(md[md[a][b]][c] == md[a][md[b][c]]
                   for a in range(8) for b in range(8) for c in range(8))
    ident_md = all(md[a][0] == a and md[0][a] == a for a in range(8))
    latin_md = all(sorted(set(row)) == list(range(8)) for row in md)
    log("structure", MOD6_monoid=assoc_md and ident_md,
        MOD6_not_latin=not latin_md)
    print(f"  MUL8 monoid: {assoc and ident}, output H="
          f"{table_output_entropy(mul)}")
    print(f"  MIN8 semilattice: {assoc_mn}, output H="
          f"{table_output_entropy(mn)}")
    print(f"  MOD6 monoid: {assoc_md and ident_md}, not-Latin: "
          f"{not latin_md}, output H="
          f"{table_output_entropy(md)}")

    # grok the phonemes
    print("\n  --- the phoneme ladder ---")
    circuits = {}
    for name, table in new_tables.items():
        for seed in (0, 1):
            r = grok_phoneme(name, table, seed=seed)
            g = r["steps"] if r["steps"] else "never"
            print(f"  {name} s{seed}: grok={g} "
                  f"test={r['test_acc']:.3f}")
            log("ladder", task=name, seed=seed,
                grok_step=r["steps"],
                test_acc=round(r["test_acc"], 4),
                exact=r["model"] is not None)
            if r["model"] is not None:
                circuits.setdefault(name, r["model"])
        save()

    # the self-composites, direct training (E32 protocol)
    print("\n  --- the self-composites (direct training) ---")
    for name, table in new_tables.items():
        for seed in (0, 1):
            r = train_composite(table, table, seed=seed)
            g = r["grok_step"] if r["grok_step"] else "never"
            print(f"  {name}({name}) s{seed}: grok={g} "
                  f"test={r['final_test']}")
            log("self_composite", task=name, seed=seed, **r)
        # the entropy features of this composite
        feats = composite_features(table, table, 8, 8)
        log("composite_features", task=name, **feats)
        print(f"  {name}({name}) features: {feats}")
        save()

    # verdicts
    rows = [r for r in RESULTS if r["kind"] == "self_composite"]
    fam = {}
    for r in rows:
        fam.setdefault(r["task"], []).append(r["grok_step"] is not None)
    mod6_wall = not any(fam.get("MOD6", [True]))
    min8_grok = all(fam.get("MIN8", [False]))
    print(f"\n  P2 (MOD6 walls — entropy is the predictor): "
          f"{'CONFIRMED' if mod6_wall else 'FALSIFIED'} "
          f"(MOD6(MOD6) grokked {sum(fam.get('MOD6', []))}/2)")
    print(f"  P3 (MIN8 free — the collapse shortcut): "
          f"{'CONFIRMED' if min8_grok else 'FALSIFIED'}")
    log("prospective_verdict",
        P2_mod6_walls=mod6_wall, P3_min8_free=min8_grok,
        fam={k: sum(v) for k, v in fam.items()})
    return fam


def main():
    t0 = time.time()
    print("E41: THE ENTROPY HYPOTHESIS — is the wall's predictor "
          "the inner's output entropy?")
    print("pre-registered seq 222\n")
    retro_rows = retrospective()
    save()
    fam = prospective()
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
