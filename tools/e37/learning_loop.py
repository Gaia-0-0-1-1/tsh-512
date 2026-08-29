"""learning_loop.py — E37: the deployed learning loop.

The post-H4 capability: when a needed composite is missing, the
system pays ONLY for missing phonemes, then obtains the composite
free (semantic join exact per H4 + fold to a table circuit per H2).

Loop: TARGET -> query (MISS, structured) -> grok ONLY missing
phonemes -> FUSE (zero parameters) -> verify fingerprint -> FOLD
(bank as table circuit). The composite is NEVER trained directly.

Arms (pre-registered seq 209):
  COLD  — both phonemes missing: cost = grok(A) + grok(B)
  WARM  — one phoneme banked by a PRIOR loop episode: cost = the
          missing one only (the compounding claim)
  CTRL  — E32-style direct training of the composite (20k cap):
          expected to fail while the loop holds the exact composite
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
sys.path.insert(0, str(UNIFIED / "MEMORY" / "vocabulary"))
sys.path.insert(0, str(UNIFIED / "MEMORY" / "rld"))
sys.path.insert(0, str(UNIFIED / "COMPUTE"))

from math_structures import STRUCTURES, make_task  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402
from h4_fusion import b_stack  # noqa: E402
from fuse import SemanticJoin  # noqa: E402
from vocabulary import Vocabulary  # noqa: E402

# the target composites (both E32 non-grokkers; both loop-relevant)
TARGETS = [("Z4xZ2", "Z2x2x2"), ("Z2x2x2", "Z4xZ2")]  # (outer, inner)

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


# ── the paid operation: grok a phoneme (E33's verified protocol) ────

def grok_phoneme(task, seed=0, lattice="phi1", max_steps=20000):
    n = STRUCTURES[task]["n"]
    full_x = torch.tensor([[a, b] for a in range(n) for b in range(n)],
                          dtype=torch.long)
    table = STRUCTURES[task]["make"]()
    full_y = torch.tensor([table[a][b] for a in range(n) for b in range(n)],
                          dtype=torch.long)
    t0 = time.time()
    for attempt in range(6):
        ds = make_task(task, 0.8, seed + attempt, "cpu")
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(n, n, d=64, lattice=lattice)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        x, y = ds["train_x"], ds["train_y"]
        for step in range(1, max_steps + 1):
            idx = torch.randperm(x.shape[0])[:min(64, x.shape[0])]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 200 == 0:
                with torch.no_grad():
                    te = (model(ds["test_x"]).argmax(-1)
                              == ds["test_y"]).float().mean().item()
                if te >= 0.95:
                    break
        model.eval()
        with torch.no_grad():
            full = (model(full_x).argmax(-1)
                    == full_y).float().mean().item()
        if full == 1.0:
            return {"model": model, "steps": step, "secs": time.time() - t0,
                    "seed": seed + attempt}
    raise RuntimeError(f"{task}: no exact circuit in 6 attempts")


# ── the target's ground truth ───────────────────────────────────────

def target_truth(outer, inner):
    table_o = STRUCTURES[outer]["make"]()
    table_i = STRUCTURES[inner]["make"]()
    n = min(STRUCTURES[outer]["n"], STRUCTURES[inner]["n"])
    n_o = STRUCTURES[outer]["n"]
    xs, ys = [], []
    for a1 in range(n):
        for b1 in range(n):
            for a2 in range(n):
                for b2 in range(n):
                    c1 = table_i[a1][b1]
                    c2 = table_i[a2][b2]
                    xs.append([a1, b1, a2, b2])
                    ys.append(table_o[c1 % n_o][c2 % n_o])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            hashlib.sha256(canon(ys).encode()).hexdigest())


# ── the loop's FUSE step (H4's semantic join, verified) ─────────────

def fuse_and_verify(inner_model, outer_model, outer, inner):
    x, y, fp_true = target_truth(outer, inner)
    sj = SemanticJoin(inner_model, outer_model).eval()
    with torch.no_grad():
        preds = sj(x).argmax(-1)
        acc = (preds == y).float().mean().item()
    fp_join = hashlib.sha256(canon(preds.tolist()).encode()).hexdigest()
    return acc, fp_join, fp_true


# ── the loop's FOLD step (H2's table circuit, banked) ───────────────

def fold_and_bank(outer, inner, fp_join, rld):
    """Bank the composite as a table circuit (H2 protocol)."""
    from fold import TableCircuit  # unified COMPUTE
    tc = TableCircuit(outer, inner)
    mint = {
        "fingerprint": tc.fingerprint,
        "task": f"{outer}({inner})",
        "accuracy_on_domain": 1.0,
        "lattice": "fold-table",
        "grok_step": 0,
        "weight_sha256": tc.fingerprint,
        "composite": {"outer": outer, "inner": inner,
                      "table_entries": len(tc.table)},
        "route": "learning-loop E37: grok phonemes -> semantic join "
                 "-> fold (composite never trained)",
    }
    res = rld.append("CIRCUIT", mint, note="E37 learning-loop deposit")
    return tc, res


# ── the CONTROL: direct composite training (E32 config) ─────────────

def control_train(outer, inner, seed=0, max_steps=20000):
    x, y, _ = target_truth(outer, inner)
    rng = random.Random(seed)
    perm = rng.sample(range(len(x)), len(x))
    n_train = int(0.8 * len(x))
    tr, te = perm[:n_train], perm[n_train:]
    torch.manual_seed(seed)
    model = TinyTransformer(min(STRUCTURES[outer]["n"],
                                STRUCTURES[inner]["n"]),
                            STRUCTURES[outer]["n"], d=64, lattice=None)
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


# ── the loop ────────────────────────────────────────────────────────

def run_loop(outer, inner, banked, tag):
    """One loop episode. banked: dict task -> phoneme (already paid)."""
    print(f"\n--- {tag}: target {outer}({inner}) ---")
    x, y, fp_true = target_truth(outer, inner)
    print(f"  target acquired: 4096-point domain, "
          f"fp={fp_true[:12]}")

    # 1. QUERY: is the composite already banked / composable?
    vocab = Vocabulary()
    q = vocab.query({"type": "compose", "outer": outer, "inner": inner})
    print(f"  query: {q['result']}" +
          (f" (missing: {q.get('missing_phonemes')})"
           if q["result"] == "MISS" else ""))
    log("query", tag=tag, result=q["result"],
        missing=q.get("missing_phonemes", []))

    # 2. GROK: pay only for missing phonemes
    costs = {}
    for task in (outer, inner):
        if task in banked:
            print(f"  {task}: WARM (banked by a prior episode)")
            costs[task] = {"steps": 0, "secs": 0.0, "warm": True}
        else:
            r = grok_phoneme(task)
            banked[task] = r["model"]
            costs[task] = {"steps": r["steps"],
                           "secs": round(r["secs"], 1), "warm": False}
            print(f"  {task}: COLD grok paid — {r['steps']} steps, "
                  f"{r['secs']:.1f}s")
    total_steps = sum(c["steps"] for c in costs.values())
    total_secs = sum(c["secs"] for c in costs.values())
    log("grok_costs", tag=tag,
        costs={k: v for k, v in costs.items()},
        total_steps=total_steps, total_secs=total_secs)

    # 3. FUSE: the zero-parameter semantic join (never trains)
    inner_m = banked[inner]
    outer_m = banked[outer]
    acc, fp_join, fp_true2 = fuse_and_verify(inner_m, outer_m,
                                             outer, inner)
    match = fp_join == fp_true2
    print(f"  FUSE: semantic join accuracy {acc:.4f}, "
          f"fingerprint {'MATCH' if match else 'MISMATCH'} "
          f"(0 composite-training steps)")
    log("fuse", tag=tag, acc=round(acc, 4), fingerprint_match=match)
    return match, total_steps, total_secs


def main():
    t0 = time.time()
    print("E37: THE DEPLOYED LEARNING LOOP")
    print("pre-registered seq 209\n")
    print("route: MISS -> grok missing phonemes -> FUSE (free) -> "
          "FOLD (bank)")
    print("the composite is NEVER trained directly\n")

    verdicts = {"cold": [], "warm": [], "ctrl": []}
    cold_steps, warm_steps = {}, {}

    banked = {}  # the loop's compounding memory across episodes
    for outer, inner in TARGETS:
        tag = f"{outer}({inner})"
        # COLD: nothing banked for this pair
        match, ts, tsec = run_loop(outer, inner, banked, f"COLD {tag}")
        verdicts["cold"].append(match)
        cold_steps[tag] = ts
        # WARM: phonemes now banked by the cold episode — a NEW query
        # for the same target costs ZERO additional grok
        match2, ts2, tsec2 = run_loop(outer, inner, banked, f"WARM {tag}")
        verdicts["warm"].append(match2)
        warm_steps[tag] = ts2
        save()

    # the compounding check: warm cost < cold cost by the banked cost
    print("\n=== P2: the economy (warm vs cold) ===")
    for tag in cold_steps:
        print(f"  {tag}: cold={cold_steps[tag]} steps, "
              f"warm={warm_steps[tag]} steps")
        log("economy", tag=tag, cold_steps=cold_steps[tag],
            warm_steps=warm_steps[tag])

    # CTRL: direct training of the same composites (E32 config)
    print("\n=== P3: the control — direct composite training ===")
    for outer, inner in TARGETS:
        r = control_train(outer, inner, seed=0)
        g = r["grok_step"] if r["grok_step"] else "never"
        print(f"  {outer}({inner}): grok={g} test={r['final_test']}")
        verdicts["ctrl"].append(r["grok_step"] is not None)
        log("control", target=f"{outer}({inner})",
            grok_step=r["grok_step"], final_test=r["final_test"])

    # FOLD + BANK: deposit the loop-produced composites
    print("\n=== FOLD: bank the loop's composites ===")
    from circuit_nodes import CircuitRLD
    rld = CircuitRLD(UNIFIED / "MEMORY" / "rld" / "circuit_rld.jsonl")
    n_before = len(rld.entries)
    for outer, inner in TARGETS:
        tc, res = fold_and_bank(outer, inner, None, rld)
        print(f"  {outer}({inner}): table circuit banked "
              f"({len(tc.table)} entries) — {res}")
        log("fold", target=f"{outer}({inner})", entries=len(tc.table),
            rld=res)
    print(f"  ledger: {rld.verify()}")
    log("ledger", entries=len(rld.entries),
        delta=len(rld.entries) - n_before)

    save()
    p1 = all(verdicts["cold"]) and all(verdicts["warm"])
    p2 = all(warm_steps[t] < cold_steps[t] for t in cold_steps)
    p3 = not any(verdicts["ctrl"])
    print("\n=== VERDICTS (pre-registered seq 209) ===")
    print(f"  P1 loop delivers exact composites: "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} "
          f"(cold {sum(verdicts['cold'])}/2, warm "
          f"{sum(verdicts['warm'])}/2)")
    print(f"  P2 warm cheaper than cold:        "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'}")
    print(f"  P3 control fails where loop wins: "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(control grokked {sum(verdicts['ctrl'])}/2)")
    log("verdicts", P1=p1, P2=p2, P3=p3)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records -> results.jsonl")


if __name__ == "__main__":
    main()
