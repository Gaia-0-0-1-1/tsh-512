"""t7_gate.py — E46: the T7 phoneme-gate mystery.

T7 = (a+b) mod 7 on domain {0..7} with vocab 8 — the dead class 7
never appears in labels. The phoneme fails 0/14 attempts while its
composite groks 4/4 and Z7 (vocab 7) groks reliably.

Arms (6 seeds each, the E43 protocol):
  1. vocab-8 unmasked   — the known failure, replicated
  2. vocab-8 masked     — loss over logits[:, :7] only
  3. vocab-7            — the Z7 baseline (drop the dead class)
  4. config relief      — unmasked vocab-8 at wd {0.1, 1.0}, cap 50k

Pre-registration: tsh-512 seq 235.
"""
import json
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


def make_T7():
    return [[(a + b) % 7 for b in range(8)] for a in range(8)]


def run_arm(name, vocab, masked, seed=0, wd=0.5, max_steps=20000):
    """Train one T7 phoneme. vocab 7 or 8; masked drops class-7
    logits from the loss."""
    table = make_T7()
    xs = [[a, b] for a in range(8) for b in range(8)]
    ys = [table[a][b] for a in range(8) for b in range(8)]
    fx = torch.tensor(xs, dtype=torch.long)
    fy = torch.tensor(ys, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(64, generator=g)
    tr, te = perm[:51], perm[51:]
    torch.manual_seed(seed)
    model = TinyTransformer(8, vocab, d=64, lattice="phi1")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=wd, betas=(0.9, 0.98))
    x_tr, y_tr = fx[tr], fy[tr]
    grok_step, te_acc, full_acc = None, 0.0, 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randperm(x_tr.shape[0])[:51]
        out = model(x_tr[idx])
        if masked:
            out = out[:, :7]
        loss = F.cross_entropy(out, y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                o = model(fx[te])
                if masked:
                    o = o[:, :7]
                te_acc = (o.argmax(-1) == fy[te]).float().mean().item()
            if te_acc >= 0.95:
                grok_step = step
                break
    model.eval()
    with torch.no_grad():
        o = model(fx)
        if masked:
            o = o[:, :7]
        full_acc = (o.argmax(-1) == fy).float().mean().item()
    return {"grok_step": grok_step, "test_acc": round(te_acc, 4),
            "full_acc": round(full_acc, 4)}


def main():
    t0 = time.time()
    print("E46: THE T7 PHONEME-GATE MYSTERY — the dead class?")
    print("pre-registered seq 235\n")

    table = make_T7()
    labels = [v for row in table for v in row]
    print(f"  T7 label support: {sorted(set(labels))} "
          f"(class 7 never appears: {7 not in set(labels)})")

    # ── arm 1-3: the dead-class test ──
    print("\n=== the dead-class test (6 seeds each) ===")
    arms = [
        ("vocab8_unmasked", dict(vocab=8, masked=False)),
        ("vocab8_masked", dict(vocab=8, masked=True)),
        ("vocab7", dict(vocab=7, masked=False)),
    ]
    arm_results = {}
    for name, kw in arms:
        grokked, exact = 0, 0
        for seed in range(6):
            r = run_arm(name, seed=seed, **kw)
            grokked += r["grok_step"] is not None
            exact += r["full_acc"] == 1.0
            g = r["grok_step"] if r["grok_step"] else "never"
            print(f"  {name} s{seed}: grok={g} "
                  f"test={r['test_acc']} full={r['full_acc']}")
            log("arm", arm=name, seed=seed, **r)
        arm_results[name] = (grokked, exact)
        print(f"  -> {name}: grokked {grokked}/6, exact {exact}/6")
        save()

    # ── arm 4: the config relief ──
    print("\n=== the config relief (unmasked vocab-8, 6 seeds) ===")
    relief = {}
    for wd in (0.1, 1.0):
        grokked, exact = 0, 0
        for seed in range(6):
            r = run_arm(f"wd{wd}", vocab=8, masked=False, seed=seed,
                        wd=wd, max_steps=50000)
            grokked += r["grok_step"] is not None
            exact += r["full_acc"] == 1.0
            g = r["grok_step"] if r["grok_step"] else "never"
            print(f"  wd={wd} s{seed}: grok={g} "
                  f"test={r['test_acc']} full={r['full_acc']}")
            log("relief", wd=wd, seed=seed, **r)
        relief[f"wd{wd}"] = (grokked, exact)
        print(f"  -> wd={wd}: grokked {grokked}/6, exact {exact}/6")
        save()

    # ── verdicts ──
    print("\n=== VERDICTS (pre-registered seq 235) ===")
    u_grok = arm_results["vocab8_unmasked"][0]
    m_grok = arm_results["vocab8_masked"][0]
    v7_grok = arm_results["vocab7"][0]
    p1 = m_grok >= 4 and u_grok <= 1 and v7_grok >= 4
    print(f"  P1 (dead class is the cause: masked {m_grok}/6 vs "
          f"unmasked {u_grok}/6, vocab7 {v7_grok}/6): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'}")
    p2 = any(v[1] >= 3 for v in relief.values())
    print(f"  P2 (config relief exists): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} {relief}")
    log("verdicts", P1_dead_class=p1, P2_config_relief=p2,
        arms={k: v for k, v in arm_results.items()}, relief=relief)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
