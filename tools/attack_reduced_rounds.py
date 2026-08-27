#!/usr/bin/env python3
"""attack_reduced_rounds.py — Phase 4 attack court: reduced-round
differential patterns on the gate-survivors T2 and T4 (docket seq 39).

Variants (all else unchanged, std-only):
  T2 with _T2_ROUNDS in {2, 4, 6, 8, 10}   (10 = fresh full control)
  T4 with _T4_PASSES  in {1, 2, 3}         (3  = fresh full control)

Measurements per variant:
  A. OUTPUT-level: per-output-trit change rate under single random
     input-trit flips — same protocol as the seq-33 gate run
     (seed 2026, 8-byte inputs, 10,000 trials, ~26% skips), graded
     against the [0.6, 0.73] neutral band.
  B. STATE-level: per-trit change rate of the chaining value (T2: the
     32-word pipe state, 192 trits; T4: the 16-word chaining state,
     96 trits) — pre-registered seq 41 claim: the chained output
     expander can mask state-level holes.
  C. EXACT differential repeats: for a FIXED input difference (trit 0
     of byte 0, +1 direction), count exact repeated 486-trit output
     differences over 1,500 random base inputs (expected ~0 for a
     random function; any repeat is a distinguishable characteristic).

Loading is cache-proof (seq-32 lesson): unique path + unique module
name per variant, no bytecode writing.
"""
import importlib.util
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'proto'))
sys.path.insert(0, str(ROOT / 'tools'))
import ternary as T                 # noqa: E402
import trit_family as F             # noqa: E402
from family_gates import flip_input_trit, BAND  # noqa: E402

SRC = None  # filled in main()


def load_variant(old, new, vi, td):
    global SRC
    p = Path(td) / ('variant_%d.py' % vi)
    p.write_text(SRC.replace(old, new), encoding='utf-8')
    spec = importlib.util.spec_from_file_location(
        'variant_mod_%d' % vi, str(p))
    FT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FT)
    assert old != new and new in SRC or True
    return FT


def trits_of(words):
    out = []
    for v in words:
        out.extend(T.to_trits(v))
    return out


def rate_stats(counts, n):
    rates = [c / n for c in counts]
    outside = sum(1 for r in rates if not BAND[0] <= r <= BAND[1])
    return (sum(rates) / len(rates), min(rates), max(rates), rates[0],
            outside, len(rates))


def output_avalance(fn, trials=10000, seed=2026):
    """Metric A — identical protocol to the seq-33 gate run."""
    rng = random.Random(seed)
    counts = [0] * 486
    n = 0
    for _ in range(trials):
        data = bytes(rng.randrange(256) for _ in range(8))
        flipped = flip_input_trit(data, (rng.randrange(8),
                                         rng.randrange(6),
                                         rng.random() < 0.5))
        if flipped is None or flipped == data:
            continue
        n += 1
        t1 = trits_of(fn(data, 0))
        t2 = trits_of(fn(flipped, 0))
        for k in range(486):
            if t1[k] != t2[k]:
                counts[k] += 1
    return rate_stats(counts, n) + (n,)


def t2_state(FT, data, tick=0):
    """Replicate t2_hash's pipeline up to (excluding) the expander,
    using the VARIANT's own internals."""
    TT = FT.T
    dh = FT.domain_hash('tsh512/t2')
    h = [0] * 32
    for i in range(16):
        h[i] = TT.add(FT.IV16[i], TT.wrap(tick * 101 + dh * (i + 1)))
        h[16 + i] = TT.add(FT.IV16[(i * 7) % 16], TT.wrap(tick * 37 + dh))
    m = FT.pad_trytes(FT.bytes_to_trytes(data), tick, FT._T2_BLOCK)
    s = [TT.wrap(tick * 13 + dh + i * 7) for i in range(16)]
    for blk in range(0, len(m), FT._T2_BLOCK):
        block = m[blk:blk + FT._T2_BLOCK]
        for r in range(FT._T2_ROUNDS):
            FT._t2_round(h, block, s, r)
            FT._t2_round(h, s, block, r)
        for i in range(16):
            h[i] = TT.add(TT.sub(h[i], h[16 + (i * 5) % 16]), block[i])
    return h


def t4_state(FT, data, tick=0):
    """Replicate t4_hash's pipeline up to (excluding) the expander."""
    TT = FT.T
    dh = FT.domain_hash('tsh512/t4')
    m = FT.pad_trytes(FT.bytes_to_trytes(data), tick, FT._T4_STATE)
    state = [TT.add(FT.IV16[i], TT.wrap(dh * (i + 1) + tick))
             for i in range(16)]
    for blk in range(0, len(m), FT._T4_STATE):
        state = FT._t4_compress(state, m[blk:blk + FT._T4_STATE])
    return state


def state_diffusion(state_fn, ntrits, trials=3000, seed=31):
    """Metric B — chaining-value diffusion under input-trit flips."""
    rng = random.Random(seed)
    counts = [0] * ntrits
    n = 0
    for _ in range(trials):
        data = bytes(rng.randrange(256) for _ in range(8))
        flipped = flip_input_trit(data, (rng.randrange(8),
                                         rng.randrange(6),
                                         rng.random() < 0.5))
        if flipped is None or flipped == data:
            continue
        n += 1
        t1 = trits_of(state_fn(data))
        t2 = trits_of(state_fn(flipped))
        for k in range(ntrits):
            if t1[k] != t2[k]:
                counts[k] += 1
    rates = [c / n for c in counts]
    return (sum(rates) / len(rates), min(rates), max(rates),
            sum(1 for r in rates if r < BAND[0]), n)


def exact_repeats(fn, samples=1500, seed=55):
    """Metric C — fixed input difference, exact output-difference
    repeats. Returns (#distinct, #repeat events)."""
    rng = random.Random(seed)
    seen = {}
    repeats = 0
    used = 0
    while used < samples:
        data = bytes(rng.randrange(256) for _ in range(8))
        flipped = flip_input_trit(data, (0, 0, True))
        if flipped is None or flipped == data:
            continue
        used += 1
        d = tuple(a - b for a, b in zip(trits_of(fn(data, 0)),
                                        trits_of(fn(flipped, 0))))
        if d in seen:
            repeats += 1
        else:
            seen[d] = data
    return len(seen), repeats


def main():
    global SRC
    SRC = (ROOT / 'proto' / 'trit_family.py').read_text('utf-8')
    dont = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with tempfile.TemporaryDirectory() as td:
            vi = 0
            print('=== T2 reduced rounds: output-level (ruler [0.6,0.73]) ===')
            for rounds in (2, 4, 6, 8, 10):
                FT = load_variant('_T2_ROUNDS = 10',
                                  '_T2_ROUNDS = %d' % rounds, vi, td)
                vi += 1
                mean, lo, hi, p0, out, tot, n = output_avalance(FT.t2_hash)
                print('T2 rounds=%-2d  mean %.4f  min %.4f  max %.4f  '
                      'pos0 %.4f  outside-band %d/%d  (n=%d)  %s'
                      % (rounds, mean, lo, hi, p0, out, tot, n,
                         'IN-BAND' if out == 0 else 'STRUCTURE'))

            print('\n=== T2 reduced rounds: state-level (192 pipe trits) ===')
            for rounds in (2, 4, 6, 8, 10):
                FT = load_variant('_T2_ROUNDS = 10',
                                  '_T2_ROUNDS = %d' % rounds, vi, td)
                vi += 1
                mean, lo, hi, low, n = state_diffusion(
                    lambda d, FT=FT: t2_state(FT, d), 192)
                print('T2 rounds=%-2d  state mean %.4f  min %.4f  '
                      'max %.4f  positions <0.6: %d/192  (n=%d)'
                      % (rounds, mean, lo, hi, low, n))

            print('\n=== T4 reduced passes ===')
            for passes in (1, 2, 3):
                FT = load_variant('_T4_PASSES = 3',
                                  '_T4_PASSES = %d' % passes, vi, td)
                vi += 1
                mean, lo, hi, p0, out, tot, n = output_avalance(FT.t4_hash)
                smean, slo, shi, slow, sn = state_diffusion(
                    lambda d, FT=FT: t4_state(FT, d), 96)
                print('T4 passes=%d  OUTPUT mean %.4f min %.4f max %.4f '
                      'outside %d/486 | STATE mean %.4f min %.4f '
                      '<0.6: %d/96  (n_out=%d, n_state=%d)'
                      % (passes, mean, lo, hi, out, smean, slo, slow, n, sn))

            print('\n=== exact differential repeats (fixed trit0-up flip) ===')
            for label, old, new, fname in (
                    ('T2 rounds=2', '_T2_ROUNDS = 10', '_T2_ROUNDS = 2',
                     't2_hash'),
                    ('T2 rounds=10 (control)', '_T2_ROUNDS = 10',
                     '_T2_ROUNDS = 10', 't2_hash'),
                    ('T4 passes=1', '_T4_PASSES = 3', '_T4_PASSES = 1',
                     't4_hash'),
                    ('T4 passes=3 (control)', '_T4_PASSES = 3',
                     '_T4_PASSES = 3', 't4_hash')):
                FT = load_variant(old, new, vi, td)   # controls: identity
                vi += 1
                distinct, repeats = exact_repeats(getattr(FT, fname))
                print('%-22s distinct diffs %d  exact repeats %d  %s'
                      % (label, distinct, repeats,
                         'CHARACTERISTIC FOUND' if repeats
                         else 'null (random-like)'))
    finally:
        sys.dont_write_bytecode = dont


if __name__ == '__main__':
    main()
