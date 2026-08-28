"""train_math.py — grokking on algebraic structures (experiment E7).

Same lab conventions (Nanda-canonical one-layer transformer), but the
task is (a, b) -> a∘b from an algebraic structure's Cayley table.
Only the structure varies — architecture and recipe are constant.

    python train_math.py --structure Z8 --precision fp --wd 0.5 \
        --seed 0 --out runs/e7/Z8_fp_wd05_s0
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from ternary_grok.model import OneLayerTransformer
from ternary_grok.telemetry import snapshot
import math_structures as MS


def evaluate(model, x, y, batch=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            logits = model(x[i:i + batch])
            correct += (logits.argmax(-1) == y[i:i + batch]).sum().item()
    model.train()
    return correct / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=list(MS.STRUCTURES))
    ap.add_argument("--precision", choices=["fp", "ternary"], required=True)
    ap.add_argument("--wd", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-mlp", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--grok-acc", type=float, default=0.95)
    ap.add_argument("--memorize-acc", type=float, default=0.99)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MS.make_task(args.structure, args.train_frac, args.seed, device)
    n = MS.STRUCTURES[args.structure]["n"]

    torch.manual_seed(args.seed)
    model = OneLayerTransformer(
        n, d_model=args.d_model, n_heads=4, d_mlp=args.d_mlp,
        ternary=args.precision == "ternary",
        seq_len=2, vocab=n, out_vocab=n,
        pos_mode="first", readout="first").to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.wd,
                            betas=(0.9, 0.98))
    batches = []
    g = torch.Generator().manual_seed(args.seed + 1)
    n_train = ds["train_x"].shape[0]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "device": device}, f, indent=2)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    milestones = {"memorize_step": None, "grok_step": None}
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        idx = torch.randperm(n_train, generator=g)[:min(512, n_train)].to(device)
        logits = model(ds["train_x"][idx])
        loss = F.cross_entropy(logits, ds["train_y"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.max_steps:
            tr_acc = evaluate(model, ds["train_x"], ds["train_y"])
            te_acc = evaluate(model, ds["test_x"], ds["test_y"])
            if milestones["memorize_step"] is None and tr_acc >= args.memorize_acc:
                milestones["memorize_step"] = step
            if milestones["grok_step"] is None and te_acc >= args.grok_acc:
                milestones["grok_step"] = step
            rec = {"step": step, "train_acc": round(tr_acc, 4),
                   "test_acc": round(te_acc, 4),
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if te_acc >= 0.99 and not milestones.get("_stopped"):
                milestones["_stopped"] = True
                break

    log.close()
    final_train = evaluate(model, ds["train_x"], ds["train_y"])
    final_test = evaluate(model, ds["test_x"], ds["test_y"])
    summary = {**milestones, "steps_run": step,
               "final_train_acc": round(final_train, 4),
               "final_test_acc": round(final_test, 4),
               "steps_per_s": round(step / max(time.time() - t0, 1e-9), 1)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(milestones), flush=True)


if __name__ == "__main__":
    main()
