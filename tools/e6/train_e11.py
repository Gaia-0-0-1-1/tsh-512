"""train_e11.py — Church multiplication with sequence outputs.

Teacher-forced seq2seq using the lab's OneLayerTransformer:
  full sequence = [input tokens (25)] + [SEP] + [output tokens (12)]
  predict every output position (26..37) with per-position CE.

For the integer-output arm, output = 1 token (the product mod 8).

Same training recipe as E7/E10 (d64, AdamW, wd 0.5, eval cadence) so
grok steps are comparable across the E-series.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from ternary_grok.model import OneLayerTransformer
from ternary_grok.telemetry import snapshot
import tasks_church_mult as TC

LAM, F, X, SEP, PAD = 0, 1, 2, 3, 4
VOCAB = 5
SEQ_LEN = 38       # 25 input + 1 SEP + 12 output
OUT_START = 26     # first output position
OUT_LEN = 12       # max Church-term length for values 0..7


def build_sequences(task):
    """Return (x, y) where x is (64, 38) token sequences and y is
    (64, 12) output-position labels (PAD-excluded mask applied later)."""
    xs, ys = [], []
    for a in range(8):
        for b in range(8):
            inp = TC.church_term(a) + [SEP] + TC.church_term(b)
            inp = (inp + [PAD] * 25)[:25]
            if task == "church_mult_z8":
                out = TC.church_term((a * b) % 8)
            elif task == "unary_mult_z8":
                out = TC.unary_term((a * b) % 8)
            elif task == "church_mult_z8_int":
                out = [(a * b) % 8]      # integer as a single token (mapped below)
            else:
                raise ValueError(task)
            out = (out + [PAD] * OUT_LEN)[:OUT_LEN]
            xs.append(inp + [SEP] + out)
            ys.append(out)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def evaluate(model, x, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(x)                     # (B, 38, VOCAB)
        out_logits = logits[:, OUT_START:, :]  # (B, 12, VOCAB)
        preds = out_logits.argmax(-1)          # (B, 12)
        # exact-match: all non-pad positions correct
        correct_pos = (preds == y) | ~mask
        exact = (correct_pos.all(dim=1)).float().mean().item()
        pos_acc = ((preds == y)[mask].float().mean().item()
                   if mask.any() else 1.0)
    model.train()
    return exact, pos_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["church_mult_z8", "church_mult_z8_int", "unary_mult_z8"])
    ap.add_argument("--precision", choices=["fp", "ternary"], required=True)
    ap.add_argument("--wd", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--grok-acc", type=float, default=0.95)
    ap.add_argument("--memorize-acc", type=float, default=0.99)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = build_sequences(args.task)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(round(args.train_frac * x.shape[0]))
    tr, te = perm[:n_train], perm[n_train:]
    x_tr, y_tr = x[tr].to(device), y[tr].to(device)
    x_te, y_te = x[te].to(device), y[te].to(device)
    mask_tr = (y_tr != PAD)
    mask_te = (y_te != PAD)

    torch.manual_seed(args.seed)
    model = OneLayerTransformer(
        VOCAB, d_model=64, n_heads=4, d_mlp=256,
        ternary=args.precision == "ternary",
        seq_len=SEQ_LEN, vocab=VOCAB, out_vocab=VOCAB,
        pos_mode="all", readout="first").to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.wd,
                            betas=(0.9, 0.98))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": device}, f, indent=2)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    milestones = {"memorize_step": None, "grok_step": None}
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        idx = torch.randperm(x_tr.shape[0],
                             generator=torch.Generator().manual_seed(
                                 args.seed * 100000 + step))[:min(64, x_tr.shape[0])]
        logits = model(x_tr[idx])                       # (B, 38, 5)
        out_logits = logits[:, OUT_START:, :]            # (B, 12, 5)
        loss = F.cross_entropy(out_logits.reshape(-1, VOCAB),
                               y_tr[idx].reshape(-1),
                               ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.max_steps:
            tr_exact, tr_pos = evaluate(model, x_tr, y_tr, mask_tr)
            te_exact, te_pos = evaluate(model, x_te, y_te, mask_te)
            if milestones["memorize_step"] is None and tr_exact >= args.memorize_acc:
                milestones["memorize_step"] = step
            if milestones["grok_step"] is None and te_exact >= args.grok_acc:
                milestones["grok_step"] = step
            rec = {"step": step, "train_exact": round(tr_exact, 4),
                   "train_pos": round(tr_pos, 4),
                   "test_exact": round(te_exact, 4),
                   "test_pos": round(te_pos, 4),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if te_exact >= 0.99:
                break

    log.close()
    tr_exact, _ = evaluate(model, x_tr, y_tr, mask_tr)
    te_exact, te_pos = evaluate(model, x_te, y_te, mask_te)
    summary = {**milestones, "steps_run": step,
               "final_train_exact": round(tr_exact, 4),
               "final_test_exact": round(te_exact, 4),
               "final_test_pos": round(te_pos, 4),
               "steps_per_s": round(step / max(time.time() - t0, 1e-9), 1)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(milestones), flush=True)


if __name__ == "__main__":
    main()
