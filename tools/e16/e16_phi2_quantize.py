"""e16_phi2_quantize.py — E16: phi2-lattice quantization of
Gemma-4-12B weights — progressive coverage, PPL per layer count.

Pre-registered at tsh-512 seq 155 (P1: phi2@48 <= 22.59 — the
sovereign's ternary-15 bake; P2: phi2 pushes coverage past 15
layers; the ternary arm at matched coverage is the control).

Lattices (from tsh-512 E20, the measured set):
  ternary {-1, 0, +1}
  phi2 (9-state Stakhov) {0, ±1, ±1/φ, ±φ, ±φ²}

Per-tensor scale: w_q = scale * nearest_level(w / scale) with
scale = max|w| / max_level (the matched-range convention).
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI = 1.618033988749895
LATTICES = {
    "ternary": [0.0, -1.0, 1.0],
    "phi2": [0.0, -1.0, 1.0, -1 / PHI, 1 / PHI,
             -PHI, PHI, -PHI * PHI, PHI * PHI],
}
MODEL = "/home/juzz/models/gemma-4-12B-hf"
OUT = Path("/home/juzz/workstation/lab/runs/e16")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "results.jsonl"
LOG = OUT / "e16.log"


def log(kind, **kw):
    rec = {"kind": kind, "t": time.time(), **kw}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"[{kind}] " + " ".join(f"{k}={v}" for k, v in kw.items()),
          flush=True)


def snap(tensor, levels):
    """Per-tensor scale-to-lattice, symmetric."""
    t = tensor.detach().float()
    mx = t.abs().max().item()
    if mx == 0:
        return tensor
    lv = torch.tensor(levels)
    top = lv.abs().max().item()
    scale = mx / top
    q = t / scale
    # nearest level (levels sorted by value for searchsorted)
    lv_sorted, _ = lv.sort()
    idx = torch.searchsorted(lv_sorted, q)
    idx = idx.clamp(1, len(lv_sorted) - 1)
    left, right = lv_sorted[idx - 1], lv_sorted[idx]
    pick = torch.where((q - left).abs() < (right - q).abs(),
                       left, right)
    return (pick * scale).to(tensor.dtype)


def quantize_layers(model, lattice, k, front=True):
    """Snap the first k (or last k) transformer layers' ALL weight
    matrices to the lattice. Returns the count of touched tensors."""
    levels = LATTICES[lattice]
    layers = model.model.layers
    order = range(k) if front else range(len(layers) - k,
                                         len(layers))
    n = 0
    with torch.no_grad():
        for i in order:
            for p in layers[i].parameters():
                if p.requires_grad_:  # all real params
                    pass
            for name, p in layers[i].named_parameters():
                p.copy_(snap(p, levels))
                n += 1
    return n


@torch.no_grad()
def ppl(model, input_ids_list, device):
    """Mean NLL per token over the calibration sequences."""
    tot_nll, tot_tok = 0.0, 0
    for ids in input_ids_list:
        ids = ids.to(device)
        out = model(ids, labels=ids)
        n_tok = (ids[:, 1:] != -100).sum().item()
        tot_nll += out.loss.float().item() * n_tok
        tot_tok += n_tok
    return tot_nll / max(tot_tok, 1)


def build_calib(tokenizer, n_seq=96, seq_len=256):
    """Deterministic wikitext-style calibration: use the tokenizer's
    own known-text heuristic — a fixed English passage tiled.
    NOTE (honest): a fixed literary passage is a convenience
    calibration, not wikitext-2; the PPL COMPARISONS are all on the
    SAME set, so deltas are internally valid; absolute numbers may
    differ from the sovereign's bake calibration. Recorded."""
    passage = (
        "The scientific method is a systematic approach to "
        "understanding the natural world through observation, "
        "hypothesis formation, experimentation, and analysis. "
        "Researchers across disciplines employ these principles "
        "to advance knowledge, from quantum physics to biology. "
        "Each experiment tests predictions, and the results "
        "either support or refute the proposed explanations, "
        "driving the iterative process of discovery forward."
    )
    ids = tokenizer(passage, return_tensors="pt").input_ids[0]
    seqs = []
    for i in range(n_seq):
        off = (i * 37) % max(len(ids) - 1, 1)
        chunk = ids[off:off + seq_len]
        if len(chunk) < seq_len:
            chunk = torch.cat([chunk, ids[:seq_len - len(chunk)]])
        seqs.append(chunk.unsqueeze(0))
    return seqs


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("start", device=device, model=MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    calib = build_calib(tok)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    log("loaded", n_layers=n_layers)

    # baseline
    base = ppl(model, calib, device)
    log("ppl", lattice="none", k=0, ppl=round(base, 4))

    schedule = [1, 4, 8, 15, 20, 30, n_layers]
    for lattice in ("phi2", "ternary"):
        # the fp snapshot lives in VRAM (100GB unified): restore
        # per k instead of disk reloads — the GPU-resident path
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16).to(device)
        model.eval()
        fp_state = {k2: v.clone() for k2, v in
                    model.state_dict().items()}
        for k in schedule:
            if k > n_layers:
                continue
            n_t = quantize_layers(model, lattice, k, front=True)
            p = ppl(model, calib, device)
            log("ppl", lattice=lattice, k=k, n_tensors=n_t,
                ppl=round(p, 4))
            # restore the fp weights in place for the next k
            with torch.no_grad():
                model.load_state_dict(fp_state)
        del model, fp_state
        torch.cuda.empty_cache()

    log("done", minutes=round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main()
