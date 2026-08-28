"""tasks_church_mult2.py — E11 fixed: next-token framing.

The lab transformer is a classifier (single readout). Frame sequence
generation as NEXT-TOKEN prediction: each (a, b) pair becomes up to 12
training rows, one per output position — the input is
[input (25) + SEP + output_prefix], the label is the NEXT output token.
The readout position (last token) carries the context; pos_mode='all'
gives every position an embedding so the model can locate itself.

This reuses the standard E10 classifier path (train_e10.py) unchanged:
make_task returns {train_x: (N, 38), train_y: (N,)} etc.

Arms:
  church_mult_z8: Church in, next-token of Church product term out
  unary_mult_z8:  unary in, next-token of unary product out
  church_mult_z8_int: Church in, integer product out (single row per pair)
"""
import torch

LAM, F, X, SEP, PAD = 0, 1, 2, 3, 4
VOCAB = 5
IN_LEN = 25
MAX_CTX = 38          # 25 + 1 SEP + up to 12 prefix tokens


def church_term(n):
    return [LAM, F, LAM, X] + [F] * n + [X]


def unary_term(n):
    return [F] * n + [X]


def next_token_rows(encoder_in, encoder_out):
    """One row per (pair, output position): context -> next token."""
    xs, ys = [], []
    for a in range(8):
        for b in range(8):
            base = encoder_in(a) + [SEP] + encoder_in(b)
            base = (base + [PAD] * IN_LEN)[:IN_LEN]
            out = encoder_out((a * b) % 8)
            for k in range(len(out)):
                ctx = base + out[:k] + [SEP]
                ctx = (ctx + [PAD] * MAX_CTX)[:MAX_CTX]
                xs.append(ctx)
                ys.append(out[k])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def int_rows(encoder_in):
    xs, ys = [], []
    for a in range(8):
        for b in range(8):
            seq = encoder_in(a) + [SEP] + encoder_in(b)
            seq = (seq + [PAD] * IN_LEN)[:IN_LEN]
            xs.append(seq)
            # map integer 0..7 into tokens: reuse F-count semantics is
            # impossible in vocab 5; instead use a separate head vocab.
            ys.append((a * b) % 8)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def make_task_e11(task, train_frac, seed, device):
    if task == "church_mult_z8":
        x, y = next_token_rows(church_term, church_term)
        out_vocab = VOCAB
    elif task == "unary_mult_z8":
        x, y = next_token_rows(unary_term, unary_term)
        out_vocab = VOCAB
    elif task == "church_mult_z8_int":
        x, y = int_rows(church_term)
        out_vocab = 8
    else:
        raise ValueError(task)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(round(train_frac * x.shape[0]))
    tr, te = perm[:n_train], perm[n_train:]
    return {"train_x": x[tr].to(device), "train_y": y[tr].to(device),
            "test_x": x[te].to(device), "test_y": y[te].to(device)}, out_vocab


def task_meta(task):
    return {"seq_len": MAX_CTX, "vocab": VOCAB}


def _verify():
    # next-token rows: correct labels, lengths
    x, y = next_token_rows(church_term, church_term)
    # 64 pairs; rows per pair = len(product term): 0*0 -> Church 0 = 5 tokens
    assert y.shape[0] > 64
    # spot: the first pair (0,0): product Church 0 = [LAM,F,LAM,X,X] (5 rows)
    first5 = y[:5].tolist()
    assert first5 == [LAM, F, LAM, X, X], first5
    # injectivity: distinct contexts
    seen = set(tuple(r.tolist()) for r in x)
    # duplicates expected across equal products — fine; check contexts
    # match their labels by regenerating
    x2, y2 = next_token_rows(unary_term, unary_term)
    assert x2.shape[1] == MAX_CTX
    # int rows
    xi, yi = int_rows(church_term)
    assert yi[0].item() == 0 and yi[63].item() == 1  # 0*0=0, 7*7=49%8=1
    assert yi.min().item() >= 0 and yi.max().item() <= 7
    return True


if __name__ == "__main__":
    print("verify:", _verify())
    for t in ("church_mult_z8", "unary_mult_z8", "church_mult_z8_int"):
        ds, ov = make_task_e11(t, 0.8, 0, "cpu")
        print("%-22s: %d train rows, out_vocab %d" % (t, ds["train_x"].shape[0], ov))
