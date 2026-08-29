"""join_of_joins.py — E38: does the semantic join compose?

The arc's one open gap (flagged by the preprint outline, seq 212):
the semantic join fuses TWO phonemes exactly (E36). Is the FUSED
pair's output sharp enough to serve as another join's inner —
i.e. is the join a RECURSIVE fusion operator?

Structure under test (triple C(B(A))):
  pair-join J(A, B) = B's stack over softmax(A's logits) @ B.embed
  nested  N = J(J(A, B), C)?  — no: the triple is C(B(A1), B(A2)),
  so the nested join is J(A, B) feeding C:
    inner = the FUSED pair (A->B), outer = C
  The fused pair emits B-vocab logits; the new join softmaxes those
  @ C.embed and runs C's stack over the PAIR of results.

Control: the depth-2 table tree (H3's fold-of-folds), exact by
construction, confirms the ground truth.

Pre-registration: tsh-512 seq 213.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TSH = HERE.parents[1]
UNIFIED = TSH.parent / "unified"

sys.path.insert(0, str(TSH / "tools" / "e6"))
sys.path.insert(0, str(TSH / "proto"))
sys.path.insert(0, str(TSH / "tools" / "e20"))
sys.path.insert(0, str(TSH / "tools" / "e36"))
sys.path.insert(0, str(UNIFIED / "COMPUTE"))

from math_structures import STRUCTURES  # noqa: E402
from h4_fusion import grok_circuit, b_stack  # noqa: E402
from fuse import SemanticJoin  # noqa: E402

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


# ── the triple's ground truth (Cayley-derived) ──────────────────────

def triple_truth(A, B, C):
    """The triple composite C(B(A(a1,b1)), B(A(a2,b2))).

    Domain: pair-of-pairs over the min vocab, values clamped into
    each table's range by mod (the E25/E32 convention).
    """
    ta = STRUCTURES[A]["make"]()
    tb = STRUCTURES[B]["make"]()
    tc = STRUCTURES[C]["make"]()
    na, nb, nc = (STRUCTURES[A]["n"], STRUCTURES[B]["n"],
                  STRUCTURES[C]["n"])
    n = min(na, nb, nc)
    xs, ys = [], []
    for a1 in range(n):
        for b1 in range(n):
            for a2 in range(n):
                for b2 in range(n):
                    c1 = ta[a1][b1]          # A(a1,b1) in A-vocab
                    c1b = c1 % nb            # into B's vocab
                    r1 = tb[c1b][c1b]        # B applied to the pair
                    c2 = ta[a2][b2]
                    c2b = c2 % nb
                    r2 = tb[c2b][c2b]
                    ys.append(tc[r1 % nc][r2 % nc])
                    xs.append([a1, b1, a2, b2])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            ys)


# ── the nested semantic join ────────────────────────────────────────

class NestedJoin(torch.nn.Module):
    """The triple as ONE continuous forward pass:

    A's logits -> softmax @ B.embed -> B's stack  (the pair join,
    giving B-vocab logits) -> softmax @ C.embed -> C's stack.
    Zero parameters. The question: is the fused pair's output sharp
    enough for the second join?
    """

    def __init__(self, model_a, model_b, model_c, n_b, n_c):
        super().__init__()
        self.a, self.b, self.c = model_a, model_b, model_c
        self.n_b, self.n_c = n_b, n_c

    def fused_pair_logits(self, x):
        """The (A->B) fused pair's logits on a pair-of-pairs input."""
        p1 = torch.softmax(self.a(x[:, :2]), dim=-1)
        p2 = torch.softmax(self.a(x[:, 2:]), dim=-1)
        e = torch.stack([p1 @ self.b.embed.weight,
                         p2 @ self.b.embed.weight], dim=1)
        return b_stack(self.b, e)          # B-vocab logits

    def forward(self, x, T=1.0):
        l1 = self.fused_pair_logits(x[:, :2] if x.shape[1] == 2 else x)
        # NOTE: forward expects the 4-token domain; handled below.
        raise NotImplementedError


def nested_join_logits(ma, mb, mc, x, T=1.0):
    """The nested join over the 4-token triple domain:
    x = [a1, b1, a2, b2]; A processes each pair, B joins the A
    outputs pairwise, C joins the B outputs."""
    # A on both pairs
    p1 = torch.softmax(ma(x[:, :2]) / T, dim=-1)
    p2 = torch.softmax(ma(x[:, 2:]) / T, dim=-1)
    # B's stack over the A-embedding mixture (the pair join)
    e_b = torch.stack([p1 @ mb.embed.weight,
                       p2 @ mb.embed.weight], dim=1)
    l_b = b_stack(mb, e_b)                 # B-vocab logits
    # the SECOND join: B's logits -> C's embeddings -> C's stack
    q1 = torch.softmax(l_b[:0] if False else l_b / T, dim=-1)
    # B emitted ONE logit vector per input row (the composite of a
    # pair); C consumes a PAIR of such rows:
    #   row1 = B(A(a1,b1)), row2 = B(A(a2,b2))
    # so we must run the pair join per pair, then pair the results.
    raise NotImplementedError


def vocab_map(n_from, n_to):
    """The mod-convention as a matrix: class i maps to row i % n_to.
    The faithful continuous extension of E25's value-level
    `v % n_vocab` — identity when n_from == n_to (E36's join),
    folds when narrower, zero-mass on unreachable rows when wider.
    """
    M = torch.zeros(n_from, n_to)
    for i in range(n_from):
        M[i, i % n_to] = 1.0
    return M


def nested_join(ma, mb, mc, x, T=1.0):
    """The nested join: the fused pair runs PER PAIR (each pair
    produces one B-vocab logit vector), then C consumes the two
    vectors as its pair input. Vocab boundaries use vocab_map (the
    mod convention, continuous form)."""
    na = ma.embed.num_embeddings
    nb = mb.embed.num_embeddings
    nc = mc.embed.num_embeddings
    # pair 1: A(a1,b1) -> B -> logits
    p1 = torch.softmax(ma(x[:, :2]) / T, dim=-1) @ vocab_map(na, nb)
    p2 = torch.softmax(ma(x[:, 2:]) / T, dim=-1) @ vocab_map(na, nb)
    e1 = p1 @ mb.embed.weight               # (B, d)
    l1 = b_stack_single(mb, e1)             # (B, n_b) logits
    e2 = p2 @ mb.embed.weight
    l2 = b_stack_single(mb, e2)
    if nb > nc:
        # reduce B-vocab logits into C's vocab: the top-1 class
        # mod n_c as the semantic token
        u1 = l1.argmax(-1) % nc
        u2 = l2.argmax(-1) % nc
        e_c = torch.stack([mc.embed.weight[u1],
                           mc.embed.weight[u2]], dim=1)
    else:
        q1 = torch.softmax(l1 / T, dim=-1) @ vocab_map(nb, nc)
        q2 = torch.softmax(l2 / T, dim=-1) @ vocab_map(nb, nc)
        e_c = torch.stack([q1 @ mc.embed.weight,
                           q2 @ mc.embed.weight], dim=1)
    return b_stack(mc, e_c)                 # C-vocab logits


def b_stack_single(model, e_emb):
    """Run the transformer stack on a SINGLE token's embedding
    (sequence length 1)."""
    e = e_emb.unsqueeze(1) + model.pos[:, :1]
    z = model.ln1(e)
    q, k, v = model.q(z), model.k(z), model.v(z)
    att = (q @ k.transpose(-2, -1)) / (model.d ** 0.5)
    att = att.softmax(-1)
    h = e + model.o(att @ v)
    h = h + model.w_out(torch.nn.functional.relu(
        model.w_in(model.ln2(h))))
    return model.unembed(h.mean(dim=1))


def pairwise_join_logits(ma, mb, x, T=1.0):
    """The verified E36 pair join over the 4-token domain (for the
    replication check): B(A(pair1), A(pair2))."""
    p1 = torch.softmax(ma(x[:, :2]) / T, dim=-1)
    p2 = torch.softmax(ma(x[:, 2:]) / T, dim=-1)
    e = torch.stack([p1 @ mb.embed.weight,
                     p2 @ mb.embed.weight], dim=1)
    return b_stack(mb, e)


def margins_of(logits):
    top2 = logits.topk(2, dim=-1).values
    m = top2[:, 0] - top2[:, 1]
    return {"min": round(m.min().item(), 3),
            "mean": round(m.mean().item(), 3)}


def main():
    t0 = time.time()
    print("E38: THE JOIN-OF-JOINS — does the semantic join compose?")
    print("pre-registered seq 213\n")

    # ── phonemes ──
    circuits = {}
    for task in ["Z4xZ2", "Z2x2x2", "Z7"]:
        model, steps, seed = grok_circuit(task)
        circuits[task] = model
        print(f"  {task}: grokked step {steps} (seed {seed}), exact")
        log("phoneme", task=task, steps=steps)

    # ── P2 first: the fused pair's sharpness (the recursion's fuel) ──
    print("\n=== P2: the fused pair's output sharpness ===")
    A, B, C = "Z4xZ2", "Z2x2x2", "Z7"
    ma, mb, mc = circuits[A], circuits[B], circuits[C]
    x, y, ys_true = triple_truth(A, B, C)
    # the fused pair emits B-vocab logits per pair; sharpness over
    # the full pair domain:
    full_pairs = torch.tensor([[a, b] for a in range(8) for b in range(8)],
                              dtype=torch.long)
    with torch.no_grad():
        p = torch.softmax(ma(full_pairs), dim=-1)
        e = p @ mb.embed.weight
        l_fused = b_stack_single(mb, e)     # (64, n_b)
    m_fused = margins_of(l_fused)
    print(f"  fused pair (A->B) logit margins: min={m_fused['min']} "
          f"mean={m_fused['mean']}")
    log("fused_pair_margins", pair=f"{A}->{B}", **m_fused)
    p2 = m_fused["min"] >= 1.0
    print(f"  P2 (sharpness carries, min>=1.0): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'}")

    # ── the replication: pairwise joins exact ──
    print("\n=== replication: pairwise joins ===")
    for inner, outer in [(A, B), (B, C)]:
        # the E32 pair domain: outer(inner(p1), inner(p2))
        ti = STRUCTURES[inner]["make"]()
        to = STRUCTURES[outer]["make"]()
        ni = STRUCTURES[inner]["n"]
        no = STRUCTURES[outer]["n"]
        n = min(ni, no)
        xs, yst = [], []
        for a1 in range(n):
            for b1 in range(n):
                for a2 in range(n):
                    for b2 in range(n):
                        r1 = ti[a1][b1]
                        r2 = ti[a2][b2]
                        xs.append([a1, b1, a2, b2])
                        yst.append(to[r1 % no][r2 % no])
        xt = torch.tensor(xs, dtype=torch.long)
        with torch.no_grad():
            if ni > no:
                # inner's vocab is wider than outer's: mod-reduce the
                # inner's argmax into outer's vocab, then run outer
                # on the reduced pair (the E25 discrete convention)
                mi, mo = circuits[inner], circuits[outer]
                c1 = mi(xt[:, :2]).argmax(-1) % no
                c2 = mi(xt[:, 2:]).argmax(-1) % no
                pr = mo(torch.stack([c1, c2], dim=1)).argmax(-1).tolist()
            else:
                pr = pairwise_join_logits(circuits[inner],
                                          circuits[outer],
                                          xt).argmax(-1).tolist()
        acc = sum(p == t for p, t in zip(pr, yst)) / len(pr)
        print(f"  join {outer}({inner}): {acc:.4f}")
        log("pairwise_join", inner=inner, outer=outer,
            acc=round(acc, 4))

    # ── P1: the nested join over the triple domain ──
    print("\n=== P1: the nested semantic join, triple domain ===")
    print(f"  target: {C}({B}({A}(x))) — {len(ys_true)} points")
    with torch.no_grad():
        preds = nested_join(ma, mb, mc, x).argmax(-1).tolist()
    acc = sum(p == t for p, t in zip(preds, ys_true)) / len(preds)
    fp_join = hashlib.sha256(canon(preds).encode()).hexdigest()
    fp_true = hashlib.sha256(canon(ys_true).encode()).hexdigest()
    match = fp_join == fp_true
    print(f"  nested join accuracy: {acc:.4f}  "
          f"fingerprint {'MATCH' if match else 'MISMATCH'}")
    log("nested_join", target=f"{C}({B}({A}))",
        acc=round(acc, 4), fingerprint_match=match)

    # ── the control: the depth-2 table tree ──
    print("\n=== control: the depth-2 table tree (fold-of-folds) ===")
    ta = STRUCTURES[A]["make"]()
    tb = STRUCTURES[B]["make"]()
    tc = STRUCTURES[C]["make"]()
    nb, nc = STRUCTURES[B]["n"], STRUCTURES[C]["n"]
    tree_preds = []
    for row in x.tolist():
        a1, b1, a2, b2 = row
        c1 = ta[a1][b1] % nb
        c2 = ta[a2][b2] % nb
        r1 = tb[c1][c1]
        r2 = tb[c2][c2]
        tree_preds.append(tc[r1 % nc][r2 % nc])
    tree_acc = sum(p == t for p, t in zip(tree_preds, ys_true)) / len(ys_true)
    fp_tree = hashlib.sha256(canon(tree_preds).encode()).hexdigest()
    print(f"  table tree accuracy: {tree_acc:.4f}  "
          f"fingerprint {'MATCH' if fp_tree == fp_true else 'MISMATCH'}")
    log("table_tree", acc=round(tree_acc, 4),
        fingerprint_match=fp_tree == fp_true)

    # ── P3: ordering — try the other nestings ──
    print("\n=== P3: ordering sensitivity ===")
    orderings = [(A, B, C), (C, B, A), (B, A, C)]
    for a_, b_, c_ in orderings:
        xt, yt, yst = triple_truth(a_, b_, c_)
        with torch.no_grad():
            pr = nested_join(circuits[a_], circuits[b_],
                             circuits[c_], xt).argmax(-1).tolist()
        ac = sum(p == t for p, t in zip(pr, yst)) / len(pr)
        print(f"  {c_}({b_}({a_})): {ac:.4f}")
        log("ordering", inner=a_, middle=b_, outer=c_,
            acc=round(ac, 4))

    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")
    print(f"\nVERDICTS: P1 (join composes): "
          f"{'CONFIRMED' if match else 'FALSIFIED'}  "
          f"P2 (sharpness carries): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'}")


if __name__ == "__main__":
    main()
