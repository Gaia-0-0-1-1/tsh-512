"""g2p.py — E55: English grapheme-to-phoneme — the reference law's
biggest-bridge test.

CMUdict's length-aligned cell: 4-letter words with exactly 4
ARPAbet phonemes (14,980 words). English orthography measured as
a PD-table (all 26 letters ambiguous in isolation).

Protocol: per-position phone classification (4 models per arm),
word-level held-out gate via chained evaluation. Train/test split
BY WORD (80/20) — the E53 lesson: no split = memorization.

Arms (2 seeds each, per position):
  G2P-RAW    view1 = letters 1-2, view2 = letters 3-4
  G2P-REF    view1 = the reference word's letters (same-rime,
             alphabetically-first in train), view2 = target letters
  G2P-RAND   random phone labels (must fail)
  G2P-FREQ   the majority-phone-per-letter control (the floor)

Pre-registration: tsh-512 seq 261.
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
LEVELS = 8       # letter quantization classes (a-z -> 0-25 clamped)


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


# ── the dataset ─────────────────────────────────────────────────────

def load_aligned():
    """4-letter words with exactly 4 phones, stress stripped."""
    words = []
    seen = set()
    for line in open(HERE / "cmudict.dict", encoding="utf-8",
                     errors="ignore"):
        parts = line.strip().split()
        if len(parts) != 5 or not parts[0].isalpha():
            continue
        w = parts[0].lower()
        if w in seen or not w.isalpha() or len(w) != 4:
            continue
        seq = [p.rstrip("012") for p in parts[1:]]
        if all(s.isalpha() and len(s) <= 2 for s in seq) and \
                len(seq) == 4:
            seen.add(w)
            words.append((w, seq))
    return words


PHONE_SET = sorted({p for _, seq in load_aligned() for p in seq})
PHONE_IDX = {p: i for i, p in enumerate(PHONE_SET)}
print(f"phone inventory: {len(PHONE_SET)}")


def letter_token(c):
    """a-z -> 0-25, clamped into the 8-level interface (mod 8 —
    the quantization convention; letters 8 apart collide, recorded
    honestly as the interface cost)."""
    return (ord(c) - ord('a')) % 8


def views_raw(word):
    return ([letter_token(c) for c in word[:2]],
            [letter_token(c) for c in word[2:]])


def views_ref(word, refword):
    return ([letter_token(c) for c in refword[:2]],
            [letter_token(c) for c in word])


def rime_key(seq):
    """the rime = last two phones (the vowel nucleus + coda)."""
    return tuple(seq[2:])


def build_reference_map(train_words):
    """For each RIME, the alphabetically-first train word — the
    deterministic canonical context (E53b's design)."""
    by_rime = defaultdict(list)
    for w, seq in train_words:
        by_rime[rime_key(seq)].append(w)
    return {rime: sorted(ws)[0] for rime, ws in by_rime.items()}


def majority_map(train_words):
    """The majority phone per (position, letter) — the FREQ arm's
    prediction table."""
    table = defaultdict(Counter)
    for w, seq in train_words:
        for i, (c, p) in enumerate(zip(w, seq)):
            table[(i, c)][p] += 1
    return {k: c.most_common(1)[0][0] for k, c in table.items()}


# ── the per-position trainer ────────────────────────────────────────

def train_position(x1_all, x2_all, y_all, te_idx, seed=0,
                   max_steps=20000):
    """Train one position model; return held-out accuracy."""
    x1 = torch.tensor(x1_all, dtype=torch.long)
    x2 = torch.tensor(x2_all, dtype=torch.long)
    y = torch.tensor(y_all, dtype=torch.long)
    te = torch.tensor(te_idx, dtype=torch.long)
    te_set = set(te_idx if isinstance(te_idx, list)
                 else te_idx.tolist())
    tr = torch.tensor([i for i in range(len(y_all))
                       if i not in te_set],
                      dtype=torch.long)
    best_te = 0.0
    k = int(max(y_all)) + 1
    for attempt in range(6):
        torch.manual_seed(seed + attempt)
        model = TinyTransformer(LEVELS, k, d=64, lattice="phi1")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=0.5, betas=(0.9, 0.98))
        for step in range(1, max_steps + 1):
            idx = tr[torch.randperm(len(tr))[:128]]
            out = model(x1[idx]) + model(x2[idx])
            loss = F.cross_entropy(out, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 500 == 0:
                with torch.no_grad():
                    te_acc = ((model(x1[te]) + model(x2[te])).argmax(-1)
                              == y[te]).float().mean().item()
                best_te = max(best_te, te_acc)
                if te_acc >= 0.99:
                    return te_acc, model
    return best_te, None


def eval_word(x1_all, x2_all, models, te_idx, words, phones_true):
    """Chained word-level evaluation: all 4 positions correct."""
    x1 = torch.tensor(x1_all, dtype=torch.long)
    x2 = torch.tensor(x2_all, dtype=torch.long)
    correct = 0
    with torch.no_grad():
        for i in te_idx:
            ok = True
            for pos in range(4):
                m = models[pos]
                if m is None:
                    ok = False
                    break
                pred = (m(x1[i:i+1]) + m(x2[i:i+1])).argmax(-1).item()
                if pred != phones_true[i][pos]:
                    ok = False
                    break
            correct += ok
    return correct / len(te_idx)


def main():
    t0 = time.time()
    print("E55: ENGLISH G2P — the reference law's biggest bridge")
    print("pre-registered seq 261\n")

    words = load_aligned()
    print(f"aligned words: {len(words)}")
    rng = random.Random(42)
    rng.shuffle(words)
    n_train = int(0.8 * len(words))
    train_words, test_words = words[:n_train], words[n_train:]
    print(f"train: {len(train_words)}, test: {len(test_words)}")

    ref_map = build_reference_map(train_words)
    maj = majority_map(train_words)
    log("dataset", total=len(words), train=len(train_words),
        test=len(test_words), phones=len(PHONE_SET),
        rimes=len(ref_map))

    # precompute per-arm datasets: for each arm, x1/x2 lists and
    # per-position labels
    def make_arm(mode):
        x1s, x2s, labels = [], [], []
        all_words = train_words + test_words
        for w, seq in all_words:
            if mode == "raw":
                v = views_raw(w)
            elif mode == "ref":
                rime = rime_key(seq)
                refw = ref_map.get(rime, w)
                v = views_ref(w, refw)
            else:
                v = views_raw(w)
            x1s.append(v[0])
            x2s.append(v[1])
            labels.append([PHONE_IDX[p] for p in seq])
        return x1s, x2s, labels

    te_idx = list(range(len(train_words), len(words)))

    arms = ["raw", "ref", "rand"]
    names = {"raw": "G2P-RAW", "ref": "G2P-REF", "rand": "G2P-RAND"}

    for mode in arms:
        x1s, x2s, labels = make_arm(mode)
        if mode == "rand":
            rr = random.Random(11)
            for lab in labels:
                for i in range(4):
                    lab[i] = rr.randrange(len(PHONE_SET))
        for seed in (0, 1):
            pos_accs, models = [], [None] * 4
            for pos in range(4):
                y = [lab[pos] for lab in labels]
                acc, m = train_position(x1s, x2s, y, te_idx,
                                        seed=seed)
                pos_accs.append(round(acc, 4))
                models[pos] = m
            word_acc = eval_word(x1s, x2s, models, te_idx,
                                 None, labels)
            mean_pos = round(sum(pos_accs) / 4, 4)
            print(f"  {names[mode]} s{seed}: "
                  f"per-position {pos_accs} mean {mean_pos}, "
                  f"word {round(word_acc, 4)}")
            log("g2p", arm=names[mode], seed=seed,
                per_position=pos_accs, mean_position=mean_pos,
                word_accuracy=round(word_acc, 4))
            save()

    # the FREQ floor (analytic, no training)
    correct = 0
    for w, seq in test_words:
        ok = all(maj.get((i, c), seq[i]) == seq[i]
                 for i, c in enumerate(w))
        correct += ok
    freq_word = correct / len(test_words)
    pos_correct = sum(1 for w, seq in test_words
                      for i, c in enumerate(w)
                      if maj.get((i, c), seq[i]) == seq[i])
    freq_pos = pos_correct / (len(test_words) * 4)
    print(f"  G2P-FREQ: per-position {round(freq_pos, 4)}, "
          f"word {round(freq_word, 4)}")
    log("g2p", arm="G2P-FREQ", per_position=round(freq_pos, 4),
        word_accuracy=round(freq_word, 4))
    save()

    # verdicts
    print("\n=== VERDICTS (pre-registered seq 261) ===")
    def meanpos(arm):
        rs = [r for r in RESULTS if r["kind"] == "g2p"
              and r["arm"] == arm]
        return round(sum(r["mean_position"] for r in rs) / len(rs),
                     4) if rs else None
    raw_m, ref_m = meanpos("G2P-RAW"), meanpos("G2P-REF")
    rand_m = meanpos("G2P-RAND")
    p1 = raw_m is not None and raw_m < 0.75
    p2 = ref_m is not None and raw_m is not None and \
        (ref_m - raw_m) >= 0.10
    p3 = rand_m is not None and rand_m < 0.15 and \
        freq_pos < raw_m
    print(f"  P1 (the PD-wall: RAW < 0.75 per-position): "
          f"{'CONFIRMED' if p1 else 'FALSIFIED'} ({raw_m})")
    print(f"  P2 (reference helps >= 10pp): "
          f"{'CONFIRMED' if p2 else 'FALSIFIED'} "
          f"(REF {ref_m} vs RAW {raw_m})")
    print(f"  P3 (controls): "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'} "
          f"(RAND {rand_m}, FREQ {round(freq_pos, 4)})")
    log("verdicts", P1_pd_wall=p1, P2_reference_helps=p2,
        P3_controls=p3, raw=raw_m, ref=ref_m, rand=rand_m,
        freq=round(freq_pos, 4))
    save()
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min — "
          f"{len(RESULTS)} records")


if __name__ == "__main__":
    main()
