"""tasks_church_mult.py — E11: Church multiplication (the composition test).

E10 made linear encodings trivial for addition (400 steps). E11 tests
COMPOSITION: church_mult = λm.λn.λf. m (n f); the product term is
f^(m·n) — up to 49 F-tokens for Z8.

Three arms (matched config):
  church_mult_z8:      Church terms in, Church TERM out (pure composition)
  church_mult_z8_int:  Church terms in, integer out (decoding isolated)
  unary_mult_z8:       unary in, unary out (binder control)

Input encoding (shared): term_a + [SEP] + term_b, padded to MAX_LEN_IN.
Output encoding:
  term-out arms: Church term of the product, padded to MAX_LEN_OUT
  int-out arm:   single token (a*b mod 8)

Input vocab: LAM=0, F=1, X=2, SEP=3, PAD=4 (5 tokens).
Term-output vocab: LAM=0, F=1, X=2, PAD=4 (4 tokens; SEP unused).
Int-output vocab: 0..7.
"""
import torch

LAM, F, X, SEP, PAD = 0, 1, 2, 3, 4
IN_VOCAB = 5
MAX_LEN_IN = 25      # 12 + 1 + 12, same as E10
MAX_LEN_OUT = 54     # 4 (binders) + 49 (F's) + 1 (X)


def church_term(n: int):
    return [LAM, F, LAM, X] + [F] * n + [X]


def unary_term(n: int):
    return [F] * n + [X]


def make_pairs(encoder_in, encoder_out):
    xs, ys = [], []
    for a in range(8):
        for b in range(8):
            seq = encoder_in(a) + [SEP] + encoder_in(b)
            seq = (seq + [PAD] * MAX_LEN_IN)[:MAX_LEN_IN]
            xs.append(seq)
            out = encoder_out((a * b) % 8)
            out = (out + [PAD] * MAX_LEN_OUT)[:MAX_LEN_OUT]
            ys.append(out)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def make_pairs_int_out(encoder_in):
    xs, ys = [], []
    for a in range(8):
        for b in range(8):
            seq = encoder_in(a) + [SEP] + encoder_in(b)
            seq = (seq + [PAD] * MAX_LEN_IN)[:MAX_LEN_IN]
            xs.append(seq)
            ys.append([(a * b) % 8])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def make_task_e11(task, train_frac, seed, device):
    if task == "church_mult_z8":
        x, y = make_pairs(church_term, church_term)
        out_vocab, out_len = 5, MAX_LEN_OUT
    elif task == "church_mult_z8_int":
        x, y = make_pairs_int_out(church_term)
        out_vocab, out_len = 8, 1
    elif task == "unary_mult_z8":
        x, y = make_pairs(unary_term, unary_term)
        out_vocab, out_len = 5, MAX_LEN_OUT
    else:
        raise ValueError(task)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(round(train_frac * x.shape[0]))
    tr, te = perm[:n_train], perm[n_train:]
    return {"train_x": x[tr].to(device), "train_y": y[tr].to(device),
            "test_x": x[te].to(device), "test_y": y[te].to(device)}, out_vocab, out_len


def task_meta(task):
    return {"seq_len": MAX_LEN_IN, "vocab": IN_VOCAB}


def _verify():
    assert church_term(7) == [LAM, F, LAM, X] + [F] * 7 + [X]
    # product terms: up to 49 F's for 7*7=49 mod 8 = 1 -> Church 1!
    # NOTE: the product is (a*b) MOD 8, so max F count is 7 (Church 7)
    # EXCEPT 7*7=49 mod 8=1 -> 1 F. The LONGEST product term is Church 7
    # (from e.g. 1*7): 4 + 7 + 1 = 12 tokens. MAX_LEN_OUT=54 is safe.
    for a in range(8):
        for b in range(8):
            p = (a * b) % 8
            assert len(church_term(p)) <= MAX_LEN_OUT
    # injectivity of inputs (shared with E10)
    x, _ = make_pairs(church_term, church_term)
    seen = set(tuple(t for t in row.tolist() if t != PAD) for row in x)
    assert len(seen) == 64
    # labels: spot checks
    _, y = make_pairs(church_term, church_term)
    assert y[0].tolist()[:len(church_term(0))] == church_term(0)      # 0*0=0
    assert y[63].tolist()[:len(church_term(1))] == church_term(1)     # 7*7=49%8=1
    xi, yi = make_pairs_int_out(church_term)
    assert yi[0].item() == 0 and yi[63].item() == 1
    return True


if __name__ == "__main__":
    print("verify:", _verify())
    for t in ("church_mult_z8", "church_mult_z8_int", "unary_mult_z8"):
        ds, ov, ol = make_task_e11(t, 0.8, 0, "cpu")
        print("%-22s: %d train, out_vocab %d, out_len %d" % (
            t, ds["train_x"].shape[0], ov, ol))
