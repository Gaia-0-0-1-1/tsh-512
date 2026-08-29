"""h1_wiring.py — E33/H1: sequential wiring — the first hypernetwork builder.

Chain two banked phoneme circuits: A's argmax output feeds B's input
(mod B's vocab, the E25 convention). Verify the wired composite's
fingerprint matches E25's derived composite fingerprint.

Pre-registration: tsh-512 seq 199.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "e6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proto"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "e20"))

from math_structures import STRUCTURES, make_task  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402

TASKS = ["Z8", "Z4xZ2", "Z2x2x2", "Z7"]


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def grok_circuit(task, seed=0, lattice="phi1", max_steps=20000):
    """Train a task to grok, return the model (the phoneme circuit)."""
    ds = make_task(task, 0.8, seed, "cpu")
    n = STRUCTURES[task]["n"]
    torch.manual_seed(seed)
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
    return model, step


def circuit_table(model, task):
    """The phoneme's truth table: all (a,b) -> output."""
    n = STRUCTURES[task]["n"]
    xs = [[a, b] for a in range(n) for b in range(n)]
    x = torch.tensor(xs, dtype=torch.long)
    with torch.no_grad():
        preds = model(x).argmax(-1).tolist()
    return preds  # index a*n+b -> output


def wire_chained(a_model, a_task, b_model, b_task):
    """Sequential wiring: A(argmax) -> mod B-vocab -> B's input.

    Returns a callable over pair-of-pairs inputs:
      ((a1,b1),(a2,b2)) -> B(A(a1,b1) mod nB, A(a2,b2) mod nB)
    """
    n_b = STRUCTURES[b_task]["n"]

    def chained(x):
        # x: tensor (B, 4) = [a1, b1, a2, b2]
        with torch.no_grad():
            # A processes the two pairs
            pair1 = x[:, :2]
            pair2 = x[:, 2:]
            c1 = a_model(pair1).argmax(-1) % n_b
            c2 = a_model(pair2).argmax(-1) % n_b
            # B processes the results as a pair
            b_in = torch.stack([c1, c2], dim=1)
            return b_model(b_in).argmax(-1)
    return chained


def composite_reference(outer, inner):
    """E25's pure composite function (the ground truth)."""
    table_o = STRUCTURES[outer]["make"]()
    table_i = STRUCTURES[inner]["make"]()
    n_o = STRUCTURES[outer]["n"]
    n_i = STRUCTURES[inner]["n"]
    n = min(n_o, n_i)

    def fn(x):
        (a1, b1), (a2, b2) = x
        c1 = table_i[a1][b1]
        c2 = table_i[a2][b2]
        return table_o[c1 % n_o][c2 % n_o]
    return fn, n


def fingerprint_of_outputs(outputs):
    return hashlib.sha256(canon(outputs).encode()).hexdigest()


def main():
    print("E33/H1: SEQUENTIAL WIRING — the first hypernetwork builder\n")
    print("chain: A(argmax) -> mod B-vocab -> B's input")
    print("verify: the wired composite's fingerprint == E25's derivation\n")

    # grok the 4 phonemes
    print("\ngrokking the 4 phonemes...")
    circuits = {}
    for task in TASKS:
        model, steps = grok_circuit(task)
        circuits[task] = model
        print(f"  {task}: grokked at step {steps}")

    # the composite domain (deterministic, 128 samples per E25's protocol)
    results = []
    print("\n--- the 16 chained composites ---")
    all_match = True
    for outer in TASKS:
        for inner in TASKS:
            ref_fn, n = composite_reference(outer, inner)
            # build the domain (E25's protocol: seed 42, 128 samples)
            rng = random.Random(42)
            dom = [((rng.randrange(n), rng.randrange(n)),
                    (rng.randrange(n), rng.randrange(n)))
                   for _ in range(128)]

            # the reference outputs (E25's function)
            ref_outs = [ref_fn(x) for x in dom]

            # the chained model's outputs
            chained = wire_chained(circuits[inner], inner,
                                   circuits[outer], outer)
            x_tensor = torch.tensor(
                [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in dom],
                dtype=torch.long)
            with torch.no_grad():
                chain_outs = chained(x_tensor).tolist()

            match = ref_outs == chain_outs
            if not match:
                all_match = False
            n_correct = sum(1 for r, c in zip(ref_outs, chain_outs)
                            if r == c)
            print(f"  {outer}({inner:<8}): "
                  f"{'MATCH' if match else f'DIFFER ({n_correct}/128)'}")
            results.append({"outer": outer, "inner": inner,
                            "match": match,
                            "accuracy": n_correct / 128})

    # the cost measurement
    print("\n--- inference cost ---")
    single = circuits["Z2x2x2"]
    x_single = torch.tensor([[3, 5]] * 64, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(500):
            single(x_single)
    single_ms = (time.time() - t0) / 500 * 1000

    chained_fn = wire_chained(circuits["Z2x2x2"], "Z2x2x2",
                               circuits["Z2x2x2"], "Z2x2x2")
    x_comp = torch.tensor([[3, 5, 1, 2]] * 64, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(500):
            chained_fn(x_comp)
    chain_ms = (time.time() - t0) / 500 * 1000

    print(f"  single circuit:  {single_ms:.2f} ms")
    print(f"  chained (2x):    {chain_ms:.2f} ms  "
          f"(ratio: {chain_ms/single_ms:.1f}x)")

    # summary
    print(f"\n=== VERDICTS ===")
    n_match = sum(1 for r in results if r["match"])
    print(f"  P1 (wiring produces correct composites): "
          f"{'CONFIRMED' if n_match == 16 else 'PARTIAL'} "
          f"({n_match}/16 exact, "
          f"{sum(r['accuracy'] for r in results)/16:.1%} avg accuracy)")
    print(f"  P2 (cost ~2x): {chain_ms/single_ms:.1f}x measured")
    print(f"  P3 (identity through the chain): "
          f"{'CONFIRMED' if all_match else 'CHECK the mismatches above'}")

    out = Path(__file__).parent / "results.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
