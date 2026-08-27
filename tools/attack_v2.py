#!/usr/bin/env python3
"""attack_v2.py — Phase 4 court re-engagement on the v2 gate-survivors
(T1-v2, T4-v2), per timeline seq 56. v1's survived-budget evidence does
NOT transfer to v2 (padding, T1 theta, and the whole T1 permutation
changed).

Variants (cache-proof loading — unique path + module name, no bytecode):
  T1-v2 with _T1_ROUNDS in {2, 4, 6}     (18 = recorded gate control)
  T4-v2 with _T4_PASSES in {1, 2}        (3  = recorded gate control)

Measurements (same instruments as the v1 court):
  A. output-level differential: per-output-trit change rate, seed 2026,
     8-byte inputs, 10,000 trials, graded on [0.6, 0.73]
  B. exact differential repeats: fixed trit0-up flip, 1,500 random base
     inputs, count exact repeated 486-trit output differences
  C. T4-v2 state-level at passes 1/2 (self-checked replication)
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

MOD = ROOT / 'proto' / 'trit_family_v2.py'
BAND = (0.6, 0.73)


def load_family(path):
    spec = importlib.util.spec_from_file_location(
        'fam_%d' % random.randrange(1 << 30), str(path.resolve()))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_variant(src, old, new, vi, td):
    p = Path(td) / ('variant_%d.py' % vi)
    p.write_text(src.replace(old, new) if old != new else src,
                 encoding='utf-8')
    return load_family(p)


def flip_input_trit(data, idx, pos, up):
    trits = T.to_trits(data[idx])
    old = trits[pos]
    new = old + 1 if up else old - 1
    if new > 1:
        new = -1
    if new < -1:
        new = 1
    trits[pos] = new
    v = T.from_trits(trits)
    if v > 255 or v < 0:
        trits[pos] = -old if old != 0 else (1 if not up else -1)
        v = T.from_trits(trits)
        if v > 255 or v < 0:
            return None
    out = bytearray(data)
    out[idx] = v
    return bytes(out)


def trits_of(words):
    out = []
    for v in words:
        out.extend(T.to_trits(v))
    return out


def output_differential(fn, trials=10000, seed=2026):
    rng = random.Random(seed)
    counts = [0] * 486
    n = 0
    for _ in range(trials):
        data = bytes(rng.randrange(256) for _ in range(8))
        flipped = flip_input_trit(data, rng.randrange(8),
                                  rng.randrange(6), rng.random() < 0.5)
        if flipped is None or flipped == data:
            continue
        n += 1
        t1 = trits_of(fn(data, 0))
        t2 = trits_of(fn(flipped, 0))
        for k in range(486):
            if t1[k] != t2[k]:
                counts[k] += 1
    rates = [c / n for c in counts]
    outside = sum(1 for r in rates if not BAND[0] <= r <= BAND[1])
    return sum(rates) / len(rates), min(rates), max(rates), outside, n


def t4_state(FT, data, tick=0):
    TT = FT.T
    dh = FT.domain_hash('tsh512/t4')
    m = FT.pad_trytes(FT.bytes_to_trytes(data), tick, FT._T4_STATE)
    state = [TT.add(FT.IV16[i], TT.wrap(dh * (i + 1) + tick))
             for i in range(16)]
    for blk in range(0, len(m), FT._T4_STATE):
        state = FT._t4_compress(state, m[blk:blk + FT._T4_STATE])
    return state


def state_differential(FT, trials=3000, seed=31):
    rng = random.Random(seed)
    counts = [0] * 96
    n = 0
    for _ in range(trials):
        data = bytes(rng.randrange(256) for _ in range(8))
        flipped = flip_input_trit(data, rng.randrange(8),
                                  rng.randrange(6), rng.random() < 0.5)
        if flipped is None or flipped == data:
            continue
        n += 1
        t1 = trits_of(t4_state(FT, data))
        t2 = trits_of(t4_state(FT, flipped))
        for k in range(96):
            if t1[k] != t2[k]:
                counts[k] += 1
    rates = [c / n for c in counts]
    return (sum(rates) / len(rates), min(rates),
            sum(1 for r in rates if r < 0.6), sum(1 for r in rates if r < 0.5))


def exact_repeats(fn, samples=1500, seed=55):
    rng = random.Random(seed)
    seen = set()
    repeats = 0
    used = 0
    while used < samples:
        data = bytes(rng.randrange(256) for _ in range(8))
        flipped = flip_input_trit(data, 0, 0, True)
        if flipped is None or flipped == data:
            continue
        used += 1
        d = tuple(a - b for a, b in zip(trits_of(fn(data, 0)),
                                        trits_of(fn(flipped, 0))))
        if d in seen:
            repeats += 1
        seen.add(d)
    return repeats


def main():
    src = MOD.read_text('utf-8')
    sys.dont_write_bytecode = True
    with tempfile.TemporaryDirectory() as td:
        vi = 0
        print('=== T1-v2 reduced rounds: output differential ===')
        for rounds in (2, 4, 6):
            FT = load_variant(src, '_T1_ROUNDS = 18',
                              '_T1_ROUNDS = %d' % rounds, vi, td)
            vi += 1
            mean, lo, hi, out, n = output_differential(FT.t1_hash)
            print('T1-v2 rounds=%-2d  mean %.4f  min %.4f  max %.4f  '
                  'outside %d/486  (n=%d)  %s'
                  % (rounds, mean, lo, hi, out, n,
                     'IN-BAND' if out == 0 else 'STRUCTURE'))

        print('\n=== T4-v2 reduced passes ===')
        for passes in (1, 2):
            FT = load_variant(src, '_T4_PASSES = 3',
                              '_T4_PASSES = %d' % passes, vi, td)
            vi += 1
            mean, lo, hi, out, n = output_differential(FT.t4_hash)
            smean, smin, s6, s5 = state_differential(FT)
            print('T4-v2 passes=%d  OUTPUT mean %.4f min %.4f outside '
                  '%d/486 | STATE mean %.4f min %.4f <0.6: %d/96 <0.5: %d/96'
                  % (passes, mean, lo, out, smean, smin, s6, s5))

        print('\n=== exact differential repeats (trit0-up, n=1500) ===')
        for label, old, new, fname in (
                ('T1-v2 rounds=2', '_T1_ROUNDS = 18', '_T1_ROUNDS = 2',
                 't1_hash'),
                ('T1-v2 rounds=18 (control)', '_T1_ROUNDS = 18',
                 '_T1_ROUNDS = 18', 't1_hash'),
                ('T4-v2 passes=1', '_T4_PASSES = 3', '_T4_PASSES = 1',
                 't4_hash'),
                ('T4-v2 passes=3 (control)', '_T4_PASSES = 3',
                 '_T4_PASSES = 3', 't4_hash')):
            FT = load_variant(src, old, new, vi, td)
            vi += 1
            r = exact_repeats(getattr(FT, fname))
            print('%-24s exact repeats %d  %s'
                  % (label, r, 'CHARACTERISTIC' if r else 'null'))


if __name__ == '__main__':
    main()
