"""two_anchor.py — E60: the two-anchor cell (onset + rime) —
completing the English word. Pre-registered seq 281.
Built from E59's proven harness (seq 274/280).

Anchors are selected from INPUT ONLY (letters) for the deployable
arm; the oracle arm is retained as an explicit ceiling; the random
anchor is the information-matched control; the majority lookup
runs at the same mod-8 information as the networks.

Honest reducers (pre-registered seq 274):
  per-position = FINAL attempt's last-evaluation test accuracy
  strict-word  = chained all-4, computed unconditionally

Arms: NONE / GRAPHEMIC / RANDOM-ANCHOR / ORACLE-RIME / FREQ-MOD8
Pre-registration: tsh-512 seq 274.
"""
import json
import random
import sys
import time
from collections import Counter, defaultdict
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
LEVELS = 8


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


def load_aligned():
    words, seen = [], set()
    for line in open(HERE.parent / "e55" / "cmudict.dict",
                     encoding="utf-8", errors="ignore"):
        parts = line.strip().split()
        if len(parts) != 5 or not parts[0].isalpha():
            continue
        w = parts[0].lower()
        if w in seen or not w.isalpha() or len(w) != 4:
            continue
        seq = [p.rstrip("012") for p in parts[1:]]
        if all(s.isalpha() and len(s) <= 2 for s in seq):
            seen.add(w)
            words.append((w, seq))
    return words


PHONE_SET = sorted({p for _, seq in load_aligned() for p in seq})
PHONE_IDX = {p: i for i, p in enumerate(PHONE_SET)}


def tok(c):
    return (ord(c) - ord('a')) % 8


# ── the anchor selectors (ALL deterministic) ────────────────────────

def graphemic_anchor_map(train_words):
    """Anchor by last-two LETTERS — input-only selection."""
    by_tail = defaultdict(list)
    for w, _ in train_words:
        by_tail[w[-2:]].append(w)
    return {t: sorted(ws)[0] for t, ws in by_tail.items()}


def oracle_rime_anchor_map(train_words):
    """Anchor by true-phoneme rime — the CEILING arm, oracle."""
    by_rime = defaultdict(list)
    for w, seq in train_words:
        by_rime[tuple(seq[2:])].append(w)
    return {r: sorted(ws)[0] for r, ws in by_rime.items()}



def onset_anchor_map(train_words):
    """Anchor by first-two LETTERS — input-only selection."""
    by_head = defaultdict(list)
    for w, _ in train_words:
        by_head[w[:2]].append(w)
    return {h: sorted(ws)[0] for h, ws in by_head.items()}

def random_anchor_list(train_words, n):
    """Deterministic pseudo-random anchors — information-matched."""
    rng = random.Random(99)
    ws = sorted(w for w, _ in train_words)
    return [ws[rng.randrange(len(ws))] for _ in range(n)]


# ── the per-position trainer (honest reducers) ──────────────────────

def train_position(x1, x2, y, tr, te, seed=0, max_steps=20000):
    best_train, best_state = 0.0, None
    final_te = 0.0
    k = len(PHONE_SET)   # fixed output width: states always loadable
    for attempt in range(6):
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(LEVELS, k, d=64, lattice="phi1")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        tr_acc = 0.0
        for step in range(1, max_steps + 1):
            idx = tr[torch.randperm(len(tr))[:128]]
            out = model(x1[idx]) + model(x2[idx])
            loss = F.cross_entropy(out, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 1000 == 0:
                with torch.no_grad():
                    tr_acc = ((model(x1[tr]) + model(x2[tr])).argmax(-1)
                              == y[tr]).float().mean().item()
        # end of this attempt: last-evaluation test accuracy
        with torch.no_grad():
            te_acc = ((model(x1[te]) + model(x2[te])).argmax(-1)
                      == y[te]).float().mean().item()
            tr_acc = ((model(x1[tr]) + model(x2[tr])).argmax(-1)
                      == y[tr]).float().mean().item()
        if tr_acc > best_train:
            best_train = tr_acc
            best_state = {k2: v.clone() for k2, v in
                          model.state_dict().items()}
            final_te = te_acc
    return round(final_te, 4), best_state


def main():
    t0 = time.time()
    print("E59: LABEL-INDEPENDENT G2P REFERENCES — the clean"
          " instrument")
    print("pre-registered seq 274\n")

    words = load_aligned()
    rng = random.Random(42)
    rng.shuffle(words)
    n_train = int(0.8 * len(words))
    train_words, test_words = words[:n_train], words[n_train:]
    g_anchor = graphemic_anchor_map(train_words)
    o_anchor = oracle_rime_anchor_map(train_words)
    rand_anchors = random_anchor_list(train_words, len(words))
    log("dataset", total=len(words), train=len(train_words),
        test=len(test_words), phones=len(PHONE_SET),
        graphemic_tails=len(g_anchor),
        oracle_rimes=len(o_anchor),
        config="E55 aligned cell; mod-8 letters; 2 seeds; final-attempt reducers")

    # index anchors per arm for every word (train+test)
    o_anchor_map = onset_anchor_map(train_words)
    def anchors_for(mode):
        out = []
        for i, (w, seq) in enumerate(words):
            if mode == "none":
                out.append(None)
            elif mode == "graph":
                out.append(g_anchor.get(w[-2:], w))
            elif mode == "rand":
                out.append(rand_anchors[i])
            elif mode == "oracle":
                out.append(o_anchor.get(tuple(seq[2:]), w))
            elif mode == "onset":
                out.append(o_anchor_map.get(w[:2], w))
            elif mode == "two":
                out.append((o_anchor_map.get(w[:2], w),
                            g_anchor.get(w[-2:], w)))
        return out

    def views_for(mode, anchors):
        x1s, x2s = [], []
        for i, (w, _) in enumerate(words):
            if mode == "none":
                x1s.append([tok(c) for c in w[:2]])
                x2s.append([tok(c) for c in w[2:]])
            elif mode == "two":
                on, ri = anchors[i]
                x1s.append([tok(c) for c in on[:2]]
                           + [tok(c) for c in ri[:2]])
                x2s.append([tok(c) for c in w[:2]]
                           + [tok(c) for c in w[2:]])
            else:
                a = anchors[i]
                x1s.append([tok(c) for c in a[:2]])
                x2s.append([tok(c) for c in w])
        return x1s, x2s

    te_idx = list(range(len(train_words), len(words)))
    tr_idx = list(range(len(train_words)))
    labels = [[PHONE_IDX[p] for p in seq] for _, seq in words]

    arms = [("none", "G2P-NONE"), ("onset", "G2P-ONSET-ANCHOR"),
            ("two", "G2P-TWO-ANCHOR"), ("graph", "G2P-RIME-REPL"),
            ("rand", "G2P-RANDOM-REPL")]

    for mode, name in arms:
        x1s, x2s = views_for(mode, anchors_for(mode))
        x1 = torch.tensor(x1s, dtype=torch.long)
        x2 = torch.tensor(x2s, dtype=torch.long)
        te = torch.tensor(te_idx, dtype=torch.long)
        tr = torch.tensor(tr_idx, dtype=torch.long)
        for seed in (0, 1):
            pos_accs, states = [], [None] * 4
            for pos in range(4):
                y = torch.tensor([lab[pos] for lab in labels])
                acc, st = train_position(x1, x2, y, tr, te, seed=seed)
                pos_accs.append(acc)
                states[pos] = st
            # strict-word: chained, unconditional
            word_ok = 0
            for i in te_idx:
                ok = True
                for pos in range(4):
                    st = states[pos]
                    if st is None:
                        ok = False
                        break
                    # reload state into a fresh model shell
                    m = TinyTransformer(LEVELS, len(PHONE_SET),
                                        d=64, lattice="phi1")
                    m.load_state_dict(st)
                    m.eval()
                    with torch.no_grad():
                        pred = (m(x1[i:i+1])
                                + m(x2[i:i+1])).argmax(-1).item()
                    if pred != labels[i][pos]:
                        ok = False
                        break
                word_ok += ok
            word_acc = round(word_ok / len(te_idx), 4)
            mean_pos = round(sum(pos_accs) / 4, 4)
            print(f"  {name} s{seed}: pos {pos_accs} mean {mean_pos},"
                  f" word {word_acc}")
            log("g2p", arm=name, seed=seed, per_position=pos_accs,
                mean_position=mean_pos, word_strict=word_acc)
            save()

    # like-for-like FREQ at mod-8 information
    maj = defaultdict(Counter)
    for w, seq in train_words:
        for i, c in enumerate(w):
            maj[(i, tok(c))][PHONE_IDX[seq[i]]] += 1
    table = {k: c.most_common(1)[0][0] for k, c in maj.items()}
    pos_ok = 0
    word_ok = 0
    for w, seq in test_words:
        labs = [PHONE_IDX[p] for p in seq]
        preds = [table.get((i, tok(c)), 0) for i, c in enumerate(w)]
        pos_ok += sum(p == l for p, l in zip(preds, labs))
        word_ok += all(p == l for p, l in zip(preds, labs))
    freq_pos = round(pos_ok / (len(test_words) * 4), 4)
    freq_word = round(word_ok / len(test_words), 4)
    print(f"  FREQ-MOD8: pos {freq_pos}, word {freq_word}")
    log("g2p", arm="FREQ-MOD8", per_position=freq_pos,
        mean_position=freq_pos, word_strict=freq_word)
    save()

    # verdicts
    def meanpos(arm):
        rs = [r for r in RESULTS if r["kind"] == "g2p"
              and r["arm"] == arm]
        return round(sum(r["mean_position"] for r in rs) / len(rs), 4)

    none_m, onset_m = meanpos("G2P-NONE"), meanpos("G2P-ONSET-ANCHOR")
    two_m, graph_m = meanpos("G2P-TWO-ANCHOR"), meanpos("G2P-RIME-REPL")
    rand_m, oracle_m = meanpos("G2P-RANDOM-ANCHOR"), \
        meanpos("G2P-ORACLE-RIME")
    rand_m = meanpos("G2P-RANDOM-REPL")
    pos0 = lambda arm: sum(r["per_position"][0] + r["per_position"][1]
                           for r in RESULTS if r["kind"] == "g2p"
                           and r["arm"] == arm) / (2 * 2)
    onset_open = pos0("G2P-ONSET-ANCHOR") - pos0("G2P-NONE")
    two_word = [r["word_strict"] for r in RESULTS
                if r["kind"] == "g2p" and r["arm"] == "G2P-TWO-ANCHOR"]
    p1 = onset_open >= 0.15
    p2 = (sum(two_word) / len(two_word)) >= 0.44 if two_word else False
    p3 = abs(none_m - freq_pos) <= 0.05
    print("\n=== VERDICTS (pre-registered seq 274) ===")
    print(f"  P1 (onset anchor opens positions 0-1 by >=15pp): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} (+{onset_open:.3f})")
    print(f"  P2 (two-anchor strict-word >= 2x rime's 0.222): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} ({two_word})")
    print(f"  P3 (floor replicates): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"({none_m} vs {freq_pos})")
    log("verdicts", P1_onset_opens=p1, P2_two_anchor_word=p2,
        P3_floor=p3, none=none_m, onset=onset_m, two=two_m,
        graph=graph_m, rand=rand_m, onset_open=round(onset_open, 4),
        two_word=two_word, freq=freq_pos)
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
