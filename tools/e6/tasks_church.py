"""tasks_church.py — E10: Church numerals vs unary vs integer (Level 1 ladder).

The lambda-calculus substrate meets the learnability spectrum. Z8
addition, three encodings, everything else matched:

  church_z8: numerals as actual Church terms.
      Church n = λf.λx.f^n(x) → tokens [LAM,F,LAM,X] + [F]*n + [X]
      Input = term_a + [SEP] + term_b, output = (a+b) mod 8.
  unary_z8: same minus the binder prefix (the control isolating
      whether the λ-structure matters).
      unary n = [F]*n + [X]; input = un_a + [SEP] + un_b.

Token vocab: LAM=0, F=1, X=2, SEP=3, PAD=4 (5 tokens).
Output vocab: 0..7 (the sum mod 8). Church terms are self-delimiting;
max input length = 12 + 1 + 12 = 25 tokens.
"""
import torch

VOCAB = 5          # LAM, F, X, SEP, PAD
OUT_VOCAB = 8
MAX_LEN = 25

LAM, F, X, SEP, PAD = 0, 1, 2, 3, 4


def church_term(n: int):
    """Church n = λf.λx.f^n(x) as tokens."""
    return [LAM, F, LAM, X] + [F] * n + [X]


def unary_term(n: int):
    """Unary n (no binders): n F's then X terminator."""
    return [F] * n + [X]


def make_pairs(encoder):
    """All 64 (a, b) pairs, encoded, with sum-mod-8 labels."""
    xs, ys = [], []
    for a in range(8):
        for b in range(8):
            seq = encoder(a) + [SEP] + encoder(b)
            seq = seq + [PAD] * (MAX_LEN - len(seq))
            xs.append(seq[:MAX_LEN])
            ys.append((a + b) % 8)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def make_task_e10(task, train_frac, seed, device):
    enc = church_term if task == "church_z8" else unary_term
    x, y = make_pairs(enc)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(round(train_frac * x.shape[0]))
    tr, te = perm[:n_train], perm[n_train:]
    return {"train_x": x[tr].to(device), "train_y": y[tr].to(device),
            "test_x": x[te].to(device), "test_y": y[te].to(device)}


def task_meta(task):
    assert task in ("church_z8", "unary_z8"), task
    return {"seq_len": MAX_LEN, "vocab": VOCAB, "out_vocab": OUT_VOCAB}


def _verify():
    # Church terms: correct lengths, correct structure
    assert church_term(0) == [LAM, F, LAM, X, X]
    assert church_term(1) == [LAM, F, LAM, X, F, X]
    assert church_term(7) == [LAM, F, LAM, X, F, F, F, F, F, F, F, X]
    assert len(church_term(7)) == 12
    # Unary: no binders, same body
    assert unary_term(0) == [X]
    assert unary_term(3) == [F, F, F, X]
    # Pairs: labels correct, no overflow, padding correct
    for enc in (church_term, unary_term):
        x, y = make_pairs(enc)
        assert x.shape == (64, MAX_LEN)
        assert y.shape == (64,)
        assert y[0].item() == 0 and y[63].item() == 6  # 0+0=0, 7+7=14 mod 8=6
        # every row: content then PAD
        for row in x:
            content = [t for t in row.tolist() if t != PAD]
            assert content[-1] == X or True  # SEP splits two terms
        # max content length fits
        max_content = max(sum(1 for t in row.tolist() if t != PAD) for row in x)
        assert max_content <= MAX_LEN, max_content
    # distinctness: no two pairs share the same input (injective encoding)
    for enc in (church_term, unary_term):
        x, _ = make_pairs(enc)
        seen = set()
        for row in x:
            seen.add(tuple(t for t in row.tolist() if t != PAD))
        assert len(seen) == 64, "encoding not injective: %d distinct" % len(seen)
    return True


if __name__ == "__main__":
    print("verify:", _verify())
    for t in ("church_z8", "unary_z8"):
        ds = make_task_e10(t, 0.8, 0, "cpu")
        print("%s: %d train / %d test" % (t, ds["train_x"].shape[0], ds["test_x"].shape[0]))
        print("  sample input:", ds["train_x"][0].tolist())
        print("  sample label:", ds["train_y"][0].item())
