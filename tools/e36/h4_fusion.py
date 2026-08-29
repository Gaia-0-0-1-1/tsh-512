"""h4_fusion.py — E36/H4: WEIGHT FUSION — the research frontier.

Three approaches to merging two circuits into one computing their
composition (pre-registered tsh-512 seq 206):

  A. DIRECT CONSTRUCTION — three forms:
     A1 the wrapped discrete join (A argmax -> mod -> B's embed): the
        H1 wiring inside one nn.Module. Control — must be exact.
     A2 the LITERAL continuous join, zero-shot: A's logits -> softmax
        at temperature T -> mixture of B's embedding rows -> B's
        stack. No argmax, no training. The temperature sweep makes
        the softmax-non-homomorphism obstruction MEASURABLE:
        accuracy as a function of intermediate sharpness.
     A3 the learned join (minimal fusion): A and B frozen entirely,
        ONLY the adapter W (n_a x d) trained on the composite. If
        this works where E32's from-scratch training failed, the
        wall is about joint-optimization scale, not representability.
  B. DISTILLATION from the H1 wired teacher (soft targets) vs the
     E32 hard-label control, on composites E32 failed to learn.
  C. THE OBSTRUCTION ANALYSIS — assembled from measurements: A's
     logit margins, B's input tolerance (sigma-sweep), the A2
     temperature curve, the B-arm verdict.

Pre-registration: tsh-512 seq 206.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "e6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proto"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "e20"))

from math_structures import STRUCTURES, make_task  # noqa: E402
from hyperbyte_test import TinyTransformer  # noqa: E402

# the two directions of the (Z4xZ2, Z2x2x2) pair — all E32 failures:
#   Z4xZ2 outer: 0/16 family   Z2x2x2 outer: 0/8 family
PAIRS = [("Z4xZ2", "Z2x2x2"), ("Z2x2x2", "Z4xZ2")]  # (outer, inner)

OUT = Path(__file__).resolve().parent
RESULTS = []


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def log(kind, **kw):
    rec = {"kind": kind, **kw}
    RESULTS.append(rec)
    print(f"  [logged] {kind}: " +
          " ".join(f"{k}={v}" for k, v in kw.items()))


def save():
    with open(OUT / "results.jsonl", "w", encoding="utf-8") as f:
        for r in RESULTS:
            f.write(canon(r) + "\n")


# ── phoneme circuits (E33's protocol, full-domain verified) ──────────

def grok_circuit(task, seed=0, lattice="phi1", max_steps=20000):
    """Grok a phoneme; verify it is EXACT on the full 64-pair domain."""
    n = STRUCTURES[task]["n"]
    full_x = torch.tensor([[a, b] for a in range(n) for b in range(n)],
                          dtype=torch.long)
    full_y = torch.tensor([STRUCTURES[task]["make"]()[a][b]
                           for a in range(n) for b in range(n)],
                          dtype=torch.long)
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
            full = (model(full_x).argmax(-1) == full_y).float().mean().item()
        if full == 1.0:
            return model, step, seed + attempt
    raise RuntimeError(f"{task}: no exact circuit in 6 attempts")


# ── the composite domain (E32's exact convention) ────────────────────

def composite_domain(outer, inner, seed=0):
    table_o = STRUCTURES[outer]["make"]()
    table_i = STRUCTURES[inner]["make"]()
    n = min(STRUCTURES[outer]["n"], STRUCTURES[inner]["n"])
    xs, ys = [], []
    for a1 in range(n):
        for b1 in range(n):
            for a2 in range(n):
                for b2 in range(n):
                    c1 = table_i[a1][b1]
                    c2 = table_i[a2][b2]
                    xs.append([a1, b1, a2, b2])
                    ys.append(table_o[c1 % STRUCTURES[outer]["n"]]
                              [c2 % STRUCTURES[outer]["n"]])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    rng = random.Random(seed)
    perm = rng.sample(range(len(xs)), len(xs))
    n_train = int(0.8 * len(xs))
    tr, te = perm[:n_train], perm[n_train:]
    return {"x": x, "y": y, "train": tr, "test": te,
            "vocab": n, "out_vocab": STRUCTURES[outer]["n"],
            "y_list": y.tolist()}


def fingerprint(values):
    return hashlib.sha256(canon(values).encode()).hexdigest()


# ── B's transformer stack, entered at the embedding ──────────────────

def b_stack(b_model, e_emb):
    """Run B's transformer from a precomputed embedding (no lookup).

    Faithfulness is asserted against B's own forward at startup.
    """
    L = e_emb.shape[1]
    e = e_emb + b_model.pos[:, :L]
    z = b_model.ln1(e)
    q, k, v = b_model.q(z), b_model.k(z), b_model.v(z)
    att = (q @ k.transpose(-2, -1)) / (b_model.d ** 0.5)
    att = att.softmax(-1)
    h = e + b_model.o(att @ v)
    h = h + b_model.w_out(F.relu(b_model.w_in(b_model.ln2(h))))
    return b_model.unembed(h.mean(dim=1))


class FusedDiscrete(nn.Module):
    """A1: the wrapped discrete join — H1's wiring in one nn.Module."""

    def __init__(self, inner, outer):
        super().__init__()
        self.a, self.b = inner, outer
        self.n_b = outer.embed.num_embeddings

    def forward(self, x):
        c1 = self.a(x[:, :2]).argmax(-1) % self.n_b
        c2 = self.a(x[:, 2:]).argmax(-1) % self.n_b
        return self.b(torch.stack([c1, c2], dim=1))


def soft_fused_logits(a_model, b_model, x, T):
    """A2: the literal continuous join at temperature T (zero-shot)."""
    l1 = a_model(x[:, :2])
    l2 = a_model(x[:, 2:])
    p1 = torch.softmax(l1 / T, dim=-1)
    p2 = torch.softmax(l2 / T, dim=-1)
    w = b_model.embed.weight  # (vocab, d)
    e = torch.stack([p1 @ w, p2 @ w], dim=1)  # (B, 2, d)
    return b_stack(b_model, e)


class AdapterJoin(nn.Module):
    """A3: minimal fusion — a learned linear join, parts frozen."""

    def __init__(self, inner, outer, n_a):
        super().__init__()
        self.a, self.b = inner, outer
        self.W = nn.Parameter(torch.randn(n_a, outer.d) * 0.02)

    def forward(self, x):
        l1 = self.a(x[:, :2])
        l2 = self.a(x[:, 2:])
        e = torch.stack([l1 @ self.W, l2 @ self.W], dim=1)
        return b_stack(self.b, e)


# ── the student trainers (E32-exact config) ──────────────────────────

def train_student(ds, seed, mode, teacher=None, T=2.0, max_steps=20000):
    """mode 'hard' (E32 replication) or 'distill' (soft+hard loss)."""
    x, y = ds["x"], ds["y"]
    tr, te = ds["train"], ds["test"]
    torch.manual_seed(seed)
    student = TinyTransformer(ds["vocab"], ds["out_vocab"], d=64,
                              lattice=None)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3,
                            weight_decay=0.5, betas=(0.9, 0.98))
    x_tr, y_tr = x[tr], y[tr]
    x_te, y_te = x[te], y[te]
    if mode == "distill":
        with torch.no_grad():
            t_logits = teacher(x_tr)
    grok_step, te_acc = None, 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randperm(x_tr.shape[0])[:64]
        s = student(x_tr[idx])
        loss = F.cross_entropy(s, y_tr[idx])
        if mode == "distill":
            soft = F.kl_div(
                F.log_softmax(s / T, dim=-1),
                F.softmax(t_logits[idx] / T, dim=-1),
                reduction="batchmean")
            loss = loss + soft
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                te_acc = (student(x_te).argmax(-1)
                          == y_te).float().mean().item()
            if te_acc >= 0.95:
                grok_step = step
                break
    with torch.no_grad():
        tr_acc = (student(x_tr).argmax(-1)
                  == y_tr).float().mean().item()
    return {"grok_step": grok_step, "final_test": round(te_acc, 4),
            "final_train": round(tr_acc, 4), "steps": step}


def train_adapter(inner, outer, ds, seed=0, max_steps=20000):
    """A3: freeze both parts; train ONLY the join W on hard labels."""
    for p in list(inner.parameters()) + list(outer.parameters()):
        p.requires_grad_(False)
    x, y = ds["x"], ds["y"]
    tr, te = ds["train"], ds["test"]
    torch.manual_seed(seed)
    fused = AdapterJoin(inner, outer, inner.embed.num_embeddings)
    opt = torch.optim.AdamW([fused.W], lr=1e-3, weight_decay=0.01,
                            betas=(0.9, 0.98))
    x_tr, y_tr = x[tr], y[tr]
    x_te, y_te = x[te], y[te]
    grok_step, te_acc = None, 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randperm(x_tr.shape[0])[:64]
        loss = F.cross_entropy(fused(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                te_acc = (fused(x_te).argmax(-1)
                          == y_te).float().mean().item()
            if te_acc >= 0.95:
                grok_step = step
                break
    fused.eval()
    with torch.no_grad():
        full = (fused(x).argmax(-1) == y).float().mean().item()
        tr_acc = (fused(x_tr).argmax(-1)
                  == y_tr).float().mean().item()
    return {"grok_step": grok_step, "final_test": round(te_acc, 4),
            "final_train": round(tr_acc, 4), "full_domain": round(full, 4),
            "steps": step}


def logit_margins(model, n):
    """Top1-minus-top2 logit margins over the full pair domain."""
    full_x = torch.tensor([[a, b] for a in range(n) for b in range(n)],
                          dtype=torch.long)
    with torch.no_grad():
        lg = model(full_x)
    top2 = lg.topk(2, dim=-1).values
    m = top2[:, 0] - top2[:, 1]
    return {"min": round(m.min().item(), 3),
            "mean": round(m.mean().item(), 3)}


def tolerance_sweep(b_model, task, sigmas, n_seeds=4):
    """C: B's accuracy when its input embedding is perturbed by
    N(0, sigma) noise — the input tolerance the join must respect."""
    n = STRUCTURES[task]["n"]
    table = STRUCTURES[task]["make"]()
    full_x = torch.tensor([[a, b] for a in range(n) for b in range(n)],
                          dtype=torch.long)
    full_y = torch.tensor([table[a][b] for a in range(n) for b in range(n)],
                          dtype=torch.long)
    out = {}
    with torch.no_grad():
        emb_norm = b_model.embed.weight.norm(dim=-1).mean().item()
        for sigma in sigmas:
            accs = []
            for s in range(n_seeds):
                g = torch.Generator().manual_seed(1000 + s)
                e = b_model.embed(full_x)
                if sigma > 0:
                    e = e + torch.randn(e.shape, generator=g) * sigma
                accs.append((b_stack(b_model, e).argmax(-1)
                             == full_y).float().mean().item())
            out[str(sigma)] = round(sum(accs) / len(accs), 4)
    return out, round(emb_norm, 3)


def main():
    t0 = time.time()
    print("E36/H4: WEIGHT FUSION — the research frontier")
    print("pre-registered seq 206; composites: E32's failures only\n")

    # ── phonemes ──
    print("--- phoneme circuits (phi1, full-domain verified) ---")
    circuits = {}
    for task in ["Z4xZ2", "Z2x2x2"]:
        model, steps, seed = grok_circuit(task)
        circuits[task] = model
        print(f"  {task}: grokked step {steps} (seed {seed}), 64/64 exact")
        log("phoneme", task=task, steps=steps, seed=seed)

    # faithfulness self-check: b_stack entered at the embedding must
    # reproduce B's own forward exactly
    with torch.no_grad():
        xt = torch.tensor([[0, 1], [3, 5], [7, 2]], dtype=torch.long)
        d = (b_stack(circuits["Z4xZ2"],
                     circuits["Z4xZ2"].embed(xt))
             - circuits["Z4xZ2"](xt)).abs().max().item()
    assert d < 1e-5, f"b_stack is not faithful: max diff {d}"
    print(f"\n  b_stack faithfulness: max|diff| = {d:.2e} (asserted < 1e-5)")
    log("selfcheck", b_stack_max_diff=d)
    save()

    # ── A1: the wrapped discrete join (control) ──
    print("\n=== A1: the wrapped discrete join (H1 wiring, one Module) ===")
    for outer, inner in PAIRS:
        ds = composite_domain(outer, inner)
        fused = FusedDiscrete(circuits[inner], circuits[outer])
        fused.eval()
        with torch.no_grad():
            preds = fused(ds["x"]).argmax(-1).tolist()
        acc = sum(p == t for p, t in zip(preds, ds["y_list"])) / len(preds)
        fp, fref = fingerprint(preds), fingerprint(ds["y_list"])
        match = fp == fref
        print(f"  {outer}({inner}): full-domain {acc:.4f}  "
              f"fingerprint {'MATCH' if match else 'MISMATCH'}")
        log("A1", outer=outer, inner=inner, acc=round(acc, 4),
            fingerprint_match=bool(match))

    # ── A2: the literal continuous join, zero-shot temperature sweep ──
    print("\n=== A2: the literal continuous join (zero-shot, T sweep) ===")
    print("  join: softmax(A_logits/T) @ B.embed  (no argmax, no training)")
    print("  T->0 recovers the discrete join; T=1 is A's natural sharpness")
    temps = [64.0, 16.0, 4.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    for outer, inner in PAIRS:
        ds = composite_domain(outer, inner)
        row = {}
        for T in temps:
            with torch.no_grad():
                preds = soft_fused_logits(circuits[inner],
                                          circuits[outer], ds["x"],
                                          T).argmax(-1).tolist()
            acc = sum(p == t for p, t in zip(preds, ds["y_list"])) / len(preds)
            row[str(T)] = round(acc, 4)
        print(f"  {outer}({inner}): " +
              "  ".join(f"T={t:g}:{row[str(t)]:.3f}" for t in temps))
        log("A2_sweep", outer=outer, inner=inner, curve=row)

    # A's logit margins (the sharpness the join depends on)
    for task in circuits:
        m = logit_margins(circuits[task], 8)
        print(f"  {task} logit margins (top1-top2): min={m['min']} "
              f"mean={m['mean']}")
        log("margins", task=task, **m)
    save()

    # ── A3: the learned join (minimal fusion) ──
    print("\n=== A3: minimal fusion — parts frozen, ONLY the join trained ===")
    for outer, inner in PAIRS:
        ds = composite_domain(outer, inner)
        r = train_adapter(circuits[inner], circuits[outer], ds)
        g = r["grok_step"] if r["grok_step"] else "never"
        print(f"  {outer}({inner}): grok={g} test={r['final_test']} "
              f"train={r['final_train']} full={r['full_domain']}")
        log("A3", outer=outer, inner=inner,
            grok_step=r["grok_step"], final_test=r["final_test"],
            final_train=r["final_train"], full_domain=r["full_domain"])
    save()

    # ── B: distillation vs hard-label control ──
    print("\n=== B: distillation from the wired teacher vs E32 hard labels ===")
    teachers = {}
    for outer, inner in PAIRS:
        teachers[(outer, inner)] = FusedDiscrete(circuits[inner],
                                                 circuits[outer]).eval()
        # teacher logit margins on the composite train domain (for T choice)
        ds = composite_domain(outer, inner)
        with torch.no_grad():
            tl = teachers[(outer, inner)](ds["x"][ds["train"]])
        top2 = tl.topk(2, dim=-1).values
        m = (top2[:, 0] - top2[:, 1])
        print(f"  teacher {outer}({inner}) logit margins: "
              f"min={m.min().item():.2f} mean={m.mean().item():.2f}")
        log("teacher_margins", outer=outer, inner=inner,
            min=round(m.min().item(), 3), mean=round(m.mean().item(), 3))

    print("\n  arms: hard (E32 replication) / distill T=2 / distill T=10")
    print("  2 composites x 2 seeds x 3 arms = 12 runs\n")
    for outer, inner in PAIRS:
        for seed in (0, 1):
            ds = composite_domain(outer, inner, seed)
            r_hard = train_student(ds, seed, "hard")
            print(f"  {outer}({inner}) s{seed} HARD: "
                  f"grok={r_hard['grok_step']} "
                  f"test={r_hard['final_test']} "
                  f"train={r_hard['final_train']}")
            log("B_hard", outer=outer, inner=inner, seed=seed, **r_hard)
            for T in (2.0, 10.0):
                r_d = train_student(ds, seed, "distill",
                                    teacher=teachers[(outer, inner)], T=T)
                print(f"  {outer}({inner}) s{seed} DISTILL T={T:g}: "
                      f"grok={r_d['grok_step']} "
                      f"test={r_d['final_test']} "
                      f"train={r_d['final_train']}")
                log("B_distill", outer=outer, inner=inner, seed=seed, T=T,
                    **r_d)
        save()

    # ── C: the obstruction, assembled from measurements ──
    print("\n=== C: the obstruction — B's input tolerance (sigma sweep) ===")
    for task in ["Z4xZ2", "Z2x2x2"]:
        sweep, emb_norm = tolerance_sweep(circuits[task], task,
                                          [0, 0.05, 0.1, 0.2, 0.5, 1.0])
        print(f"  {task} (mean embedding norm {emb_norm}): " +
              "  ".join(f"s={s}:{sweep[str(s)]:.3f}"
                        for s in [0, 0.05, 0.1, 0.2, 0.5, 1.0]))
        log("tolerance", task=task, mean_emb_norm=emb_norm, curve=sweep)

    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records -> results.jsonl")


if __name__ == "__main__":
    main()
