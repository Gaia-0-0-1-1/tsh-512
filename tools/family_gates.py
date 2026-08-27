#!/usr/bin/env python3
"""family_gates.py — IGNITION PHASE 3 gates for the T1–T4 family.

Parameterized (v1 default; v2 via --mod/--vec):

  python tools/family_gates.py freeze  [--mod ... --vec ...]
  python tools/family_gates.py gates  [--mod ... --vec ...]
  python tools/family_gates.py tamper [--mod ... --vec ...]
  python tools/family_gates.py state  [--mod ... --vec ...]

Pre-registered gates (SPEC §3, IGNITION §4), run mechanically:

  determinism  — frozen vectors recompute byte-identically
  avalanche    — flip one INPUT trit; every output trit's change rate
                 must lie in [0.6, 0.73] over >= 10,000 samples
                 (balanced ternary neutral = 2/3; NEVER the binary
                 0.5 ruler)
  diffusion    — measured reach: union of changed output positions
                 over single-trit flips must cover all 486 trits
  tamper       — flip constants; every tamper must make the frozen
                 vectors fail (fixture-level detection)
  state        — STATE-LEVEL diffusion (added at v2, pre-registered
                 seq 51 after seq 43/44 showed the output ruler is a
                 blind witness for expander-masked constructions):
                 the chaining value that feeds the output expander
                 must have mean change rate in [0.6, 0.73] AND every
                 state trit >= 0.5, n=3000 flips. The pipeline
                 replication SELF-CHECKS against the module's own
                 hash output (expand(replicated state) == hash) so a
                 drifting replication fails loudly instead of
                 measuring the wrong thing.

Vectors freeze to canonical JSON (sorted keys, no whitespace).
"""
import argparse
import importlib.util
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

args = None          # set in main()
F = None             # family module under test
VEC_PATH = None      # vectors fixture path
CONSTRUCTIONS = {}   # {'T1': fn, ...}

AVALANCHE_TRIALS = 10000
AVALANCHE_INPUT_BYTES = 8   # single-block domain for T1's rate;
                            # SPEC fixes no input size; recorded here
BAND = (0.6, 0.73)
STATE_TRIALS = 3000
STATE_FLOOR = 0.5

VEC_CASES = [
    ('empty', b'', 0),
    ('one-trit', b'\x01', 0),
    ('abc', b'abc', 0),
    ('tick0', b'abc', 0),
    ('tick1', b'abc', 1),
    ('tick2', b'abc', 2),
    ('len32', bytes(range(32)), 0),
    ('len64', bytes(range(64)), 0),
    ('len128', bytes(range(128)), 0),
    ('len729', bytes((i * 37) % 256 for i in range(729)), 0),
]


def load_family(path: Path):
    spec = importlib.util.spec_from_file_location('family_under_test',
                                                  str(path.resolve()))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def flip_input_trit(data, bit_of_entropy):
    """Change one balanced trit of the tryte encoding of one byte.

    Byte b encodes as tryte wrap(b); pick a trit position 0..5 and
    change that trit to one of its two other values; re-encode. If the
    re-encoded tryte is out of byte range 0..255, take the other new
    trit value.
    """
    idx, pos, up = bit_of_entropy
    trits = F.T.to_trits(data[idx])
    old = trits[pos]
    new = old + 1 if up else old - 1
    if new > 1:
        new = -1
    if new < -1:
        new = 1
    trits[pos] = new
    v = F.T.from_trits(trits)
    if v > 255 or v < 0:
        trits[pos] = -old if old != 0 else (1 if not up else -1)
        v = F.T.from_trits(trits)
        if v > 255 or v < 0:
            return None
    out = bytearray(data)
    out[idx] = v
    return bytes(out)


def freeze():
    doc = {'v': 1, 'law': 'IGNITION PHASE 2/3 - family vectors',
           'module': str(args.mod),
           'output': '81 trytes (486 trits), serialized as trit string',
           'vectors': []}
    for name, data, tick in VEC_CASES:
        row = {'name': name, 'input_hex': data.hex(), 'tick': tick}
        for tag, fn in CONSTRUCTIONS.items():
            row[tag] = F.output_string(fn(data, tick))
        doc['vectors'].append(row)
    VEC_PATH.parent.mkdir(exist_ok=True)
    with open(VEC_PATH, 'w', encoding='utf-8', newline='\n') as f:
        f.write(canon(doc) + '\n')
    print('froze %d family vectors x %d constructions -> %s'
          % (len(doc['vectors']), len(CONSTRUCTIONS), VEC_PATH))


def load_vectors():
    return json.loads(VEC_PATH.read_text('utf-8'))


def gate_determinism():
    vecs = load_vectors()['vectors']
    ok = True
    for row in vecs:
        data = bytes.fromhex(row['input_hex'])
        for tag, fn in CONSTRUCTIONS.items():
            if F.output_string(fn(data, row['tick'])) != row[tag]:
                print('DETERMINISM FAIL: %s %s' % (tag, row['name']))
                ok = False
    print('[determinism] %s (all %d vectors x %d recomputed identically)'
          % ('PASS' if ok else 'FAIL', len(vecs), len(CONSTRUCTIONS)))
    return ok


def _trits_of(words):
    out = []
    for v in words:
        out.extend(F.T.to_trits(v))
    return out


def gate_avalanche(only=None):
    rng = random.Random(2026)
    all_ok = True
    for tag, fn in CONSTRUCTIONS.items():
        if only and tag != only:
            continue
        counts = [0] * 486
        n_valid = 0
        t0 = time.time()
        for _ in range(AVALANCHE_TRIALS):
            data = bytes(rng.randrange(256)
                         for _ in range(AVALANCHE_INPUT_BYTES))
            flipped = flip_input_trit(data, (rng.randrange(
                AVALANCHE_INPUT_BYTES), rng.randrange(6),
                rng.random() < 0.5))
            if flipped is None or flipped == data:
                continue
            n_valid += 1
            t1 = _trits_of(fn(data, 0))
            t2 = _trits_of(fn(flipped, 0))
            for k in range(486):
                if t1[k] != t2[k]:
                    counts[k] += 1
        rates = [c / n_valid for c in counts]
        outside = sum(1 for r in rates if not BAND[0] <= r <= BAND[1])
        mean = sum(rates) / len(rates)
        status = 'PASS' if outside == 0 else 'FAIL'
        if outside:
            all_ok = False
        print('[avalanche %s] n_valid=%d  mean %.4f  min %.4f  max %.4f  '
              'pos0 %.4f  positions outside [0.6,0.73]: %d/486  %s  (%.0fs)'
              % (tag, n_valid, mean, min(rates), max(rates), rates[0],
                 outside, status, time.time() - t0))
    return all_ok


def gate_diffusion(only=None):
    """Measured reach: union of changed output positions over many
    single-trit flips must cover all 486 output trits."""
    rng = random.Random(77)
    all_ok = True
    for tag, fn in CONSTRUCTIONS.items():
        if only and tag != only:
            continue
        covered = set()
        trials = 0
        while len(covered) < 486 and trials < 400:
            data = bytes(rng.randrange(256)
                         for _ in range(AVALANCHE_INPUT_BYTES))
            flipped = flip_input_trit(data, (rng.randrange(
                AVALANCHE_INPUT_BYTES), rng.randrange(6),
                rng.random() < 0.5))
            if flipped is None or flipped == data:
                continue
            trials += 1
            h1 = fn(data, 0)
            h2 = fn(flipped, 0)
            for wi, (a, b) in enumerate(zip(h1, h2)):
                ta = F.T.to_trits(a)
                tb = F.T.to_trits(b)
                for k in range(6):
                    if ta[k] != tb[k]:
                        covered.add(wi * 6 + k)
        ok = len(covered) == 486
        if not ok:
            all_ok = False
        print('[diffusion %s] %d/486 output trits reachable by '
              'single-trit flips in %d trials  %s'
              % (tag, len(covered), trials, 'PASS' if ok else 'FAIL'))
    return all_ok


# ── state-level gate (v2, pre-registered seq 51) ────────────────────

def t1_state(data, tick=0):
    """Sponge state after absorb (the value the squeeze reads)."""
    seed = F.T.wrap(tick * 13 + F.domain_hash('tsh512/t1'))
    state = [0] * 27
    padded = F.pad_trytes(F.bytes_to_trytes(data), tick, F._T1_RATE)
    for blk in range(0, len(padded), F._T1_RATE):
        for i in range(F._T1_RATE):
            state[i] = F.T.add(state[i], padded[blk + i])
        state = F._t1_permute(state, seed)
    return state


def t2_state(data, tick=0):
    """Both pipes after all blocks (feeds the expander's h[:16])."""
    TT = F.T
    dh = F.domain_hash('tsh512/t2')
    h = [0] * 32
    for i in range(16):
        h[i] = TT.add(F.IV16[i], TT.wrap(tick * 101 + dh * (i + 1)))
        h[16 + i] = TT.add(F.IV16[(i * 7) % 16], TT.wrap(tick * 37 + dh))
    m = F.pad_trytes(F.bytes_to_trytes(data), tick, F._T2_BLOCK)
    s = [TT.wrap(tick * 13 + dh + i * 7) for i in range(16)]
    for blk in range(0, len(m), F._T2_BLOCK):
        block = m[blk:blk + F._T2_BLOCK]
        for r in range(F._T2_ROUNDS):
            F._t2_round(h, block, s, r)
            F._t2_round(h, s, block, r)
        for i in range(16):                     # v3 cross-mix
            a, b = h[i], h[16 + i]
            h[i] = TT.add(a, F.rotl_tryte(b, 1 + i % 5))
            h[16 + i] = TT.add(b, F.rotl_tryte(a, 3 + i % 4))
        for i in range(16):
            h[i] = TT.add(TT.sub(h[i], h[16 + (i * 5) % 16]), block[i])
            h[16 + i] = TT.add(TT.sub(h[16 + i], h[(i * 3) % 16]),
                               block[(i * 7 + 3) % 16])
    return h


def t3_state(data, tick=0):
    """L+R after 16 Feistel rounds (feeds the expander)."""
    TT = F.T
    dh = F.domain_hash('tsh512/t3')
    m = F.pad_trytes(F.bytes_to_trytes(data), tick, 2)
    L = [0] * F._T3_HALF
    R = [0] * F._T3_HALF
    for i, v in enumerate(m):
        R[i % F._T3_HALF] = TT.add(R[i % F._T3_HALF], v)
        if i % F._T3_HALF == F._T3_HALF - 1:
            L, R = R, L
    k = TT.wrap(tick * 101 + dh * 7 + 5)
    keys = []
    for r in range(F._T3_ROUNDS):
        k = TT.wrap(k * 37 + r * 11 + 3)
        keys.append(k)
    for r in range(F._T3_ROUNDS):
        kr = keys[r]
        a1 = TT.to_trits(TT.wrap(kr * 5 + 7))
        f = []
        for i in range(F._T3_HALF):
            x = TT.to_trits(R[i])
            a2 = TT.to_trits(TT.add(kr, i))
            a0 = TT.to_trits(TT.wrap(kr * 11 + i * 3 + 1))
            xx = F.f36_mul(x, x)
            t = F.f36_mul(a2, xx)
            t = F.f36_add(t, F.f36_mul(a1, x))
            t = F.f36_add(t, a0)
            t = F.f36_add(t, TT.to_trits(R[(i + 2) % F._T3_HALF]))
            f.append(TT.from_trits(list(t)))
        newR = [TT.add(L[i], f[i]) for i in range(F._T3_HALF)]
        L = R
        R = newR
    return L + R


def t4_state(data, tick=0):
    """Chaining state after all blocks (feeds the expander)."""
    TT = F.T
    dh = F.domain_hash('tsh512/t4')
    m = F.pad_trytes(F.bytes_to_trytes(data), tick, F._T4_STATE)
    state = [TT.add(F.IV16[i], TT.wrap(dh * (i + 1) + tick))
             for i in range(16)]
    for blk in range(0, len(m), F._T4_STATE):
        state = F._t4_compress(state, m[blk:blk + F._T4_STATE])
    return state


def _selfcheck_state_replication():
    """The replications must reproduce the module's own outputs, or the
    state gate measures the wrong thing. Fail loudly."""
    ok = True
    for data in (b'', b'abc'):
        if t1_state(data, 0)[:9] != F.t1_hash(data, 0)[:9]:
            print('STATE REPLICATION ERROR: T1'); ok = False
        if F._expand_to_81(t2_state(data, 0)[:16], 17) != F.t2_hash(data, 0):
            print('STATE REPLICATION ERROR: T2'); ok = False
        if F._expand_to_81(t3_state(data, 0), 29) != F.t3_hash(data, 0):
            print('STATE REPLICATION ERROR: T3'); ok = False
        if F._expand_to_81(t4_state(data, 0), 43) != F.t4_hash(data, 0):
            print('STATE REPLICATION ERROR: T4'); ok = False
    return ok


def gate_state():
    """Pre-registered ruler (seq 51): mean in [0.6, 0.73] AND every
    state trit rate >= 0.5, n=3000 single-trit flips."""
    if not _selfcheck_state_replication():
        return False
    rng = random.Random(31)
    all_ok = True
    states = {'T1': (t1_state, 162), 'T2': (t2_state, 192),
              'T3': (t3_state, 324), 'T4': (t4_state, 96)}
    for tag, (sfn, ntrits) in states.items():
        counts = [0] * ntrits
        n = 0
        t0 = time.time()
        for _ in range(STATE_TRIALS):
            data = bytes(rng.randrange(256)
                         for _ in range(AVALANCHE_INPUT_BYTES))
            flipped = flip_input_trit(data, (rng.randrange(
                AVALANCHE_INPUT_BYTES), rng.randrange(6),
                rng.random() < 0.5))
            if flipped is None or flipped == data:
                continue
            n += 1
            t1 = _trits_of(sfn(data))
            t2 = _trits_of(sfn(flipped))
            for k in range(ntrits):
                if t1[k] != t2[k]:
                    counts[k] += 1
        rates = [c / n for c in counts]
        mean = sum(rates) / len(rates)
        below = sum(1 for r in rates if r < STATE_FLOOR)
        below6 = sum(1 for r in rates if r < 0.6)
        ok = (BAND[0] <= mean <= BAND[1]) and below == 0
        if not ok:
            all_ok = False
        print('[state %s] mean %.4f  min %.4f  max %.4f  below %.1f: '
              '%d/%d  (below 0.6: %d)  %s  (%.0fs, n=%d)'
              % (tag, mean, min(rates), max(rates), STATE_FLOOR, below,
                 ntrits, below6, 'PASS' if ok else 'FAIL',
                 time.time() - t0, n))
    return all_ok


def _recompute_all():
    """Current module -> {tag: {vecname: digest}} for all frozen vectors."""
    vecs = load_vectors()['vectors']
    return {tag: {row['name']: F.output_string(
        fn(bytes.fromhex(row['input_hex']), row['tick']))
        for row in vecs} for tag, fn in CONSTRUCTIONS.items()}


def gate_tamper():
    """Flip constants; every tamper must change >=1 frozen vector of the
    affected constructions (fixture-level detection, as in Phase 0)."""
    src = Path(args.mod).read_text('utf-8')
    frozen = _recompute_all()

    tampers = []
    for i in range(16):
        c = F._TSH_ROUNDS[i]
        lit = '0x%016X' % c          # the source spells these uppercase
        tampers.append(('IV16[%d] ^= 1' % i,
                        lit, '0x%016X' % (c ^ 1), ('T2', 'T4')))
    tampers.append(('theta[0][1] 1 -> 2',
                    '_T1_THETA = ((0, 1, 1),',
                    '_T1_THETA = ((0, 2, 1),', ('T1',)))
    tampers.append(('expander seed 17 -> 18',
                    '_expand_to_81(h[:16], 17)',
                    '_expand_to_81(h[:16], 18)', ('T2',)))
    tampers.append(('expander seed 29 -> 30',
                    '_expand_to_81(L + R, 29)',
                    '_expand_to_81(L + R, 30)', ('T3',)))
    tampers.append(('expander seed 43 -> 44',
                    '_expand_to_81(state, 43)',
                    '_expand_to_81(state, 44)', ('T4',)))
    tampers.append(('pad marker +13 -> +14',
                    "list(msg_trytes) + [13]",
                    "list(msg_trytes) + [14]", ('T1', 'T2', 'T3', 'T4')))
    tampers.append(('T2 G rot 3 -> 4',
                    'rotl_tryte(T.sub(vd, va), 3)',
                    'rotl_tryte(T.sub(vd, va), 4)', ('T2',)))
    tampers.append(('T3 key chain 37 -> 38',
                    'k = T.wrap(k * 37 + r * 11 + 3)',
                    'k = T.wrap(k * 38 + r * 11 + 3)', ('T3',)))
    tampers.append(('T4 passes 3 -> 4',
                    'for _ in range(_T4_PASSES):',
                    'for _ in range(4):', ('T4',)))
    tampers.append(('T1 rounds 18 -> 17',
                    'for r in range(_T1_ROUNDS):',
                    'for r in range(17):', ('T1',)))

    detected = 0
    holes = []
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True   # never trust __pycache__ here: a
    # same-length rewrite of one path can silently reuse a stale .pyc
    # and execute the PREVIOUS tamper (found the hard way, seq 32)
    try:
        with tempfile.TemporaryDirectory() as td:
            for ti, (label, old, new, affects) in enumerate(tampers):
                if old not in src:
                    print('TAMPER TARGET MISSING: %r' % old)
                    holes.append(label)
                    continue
                # unique PATH and unique module name per tamper
                p = Path(td) / ('tampered_%d.py' % ti)
                p.write_text(src.replace(old, new), encoding='utf-8')
                spec = importlib.util.spec_from_file_location(
                    'tampered_mod_%d' % ti, str(p))
                FT = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(FT)
                changed = 0
                vecs = load_vectors()['vectors']
                for tag in affects:
                    fn = getattr(FT, {'T1': 't1_hash', 'T2': 't2_hash',
                                      'T3': 't3_hash',
                                      'T4': 't4_hash'}[tag])
                    for row in vecs:
                        if F.output_string(
                                fn(bytes.fromhex(row['input_hex']),
                                   row['tick'])) != frozen[tag][row['name']]:
                            changed += 1
                if changed:
                    detected += 1
                else:
                    holes.append(label)
                    print('TAMPER INVISIBLE: %s' % label)
    finally:
        sys.dont_write_bytecode = dont_write
    print('[tamper] %d/%d tampers detected by the frozen fixture %s'
          % (detected, len(tampers), 'PASS' if not holes else 'FAIL'))
    return not holes


def main():
    global args, F, VEC_PATH, CONSTRUCTIONS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('freeze', 'gates', 'tamper', 'state'))
    ap.add_argument('--mod', default=str(ROOT / 'proto' / 'trit_family.py'))
    ap.add_argument('--vec', default=str(
        ROOT / 'vectors' / 'trit_family_vectors.json'))
    ap.add_argument('--only', default=None,
                    help='restrict avalanche/diffusion to one tag (T1-T4)')
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / 'proto'))
    sys.dont_write_bytecode = True      # cache-proof module loading
    F = load_family(Path(args.mod))
    VEC_PATH = Path(args.vec)
    CONSTRUCTIONS = {'T1': F.t1_hash, 'T2': F.t2_hash,
                     'T3': F.t3_hash, 'T4': F.t4_hash}
    if args.cmd == 'freeze':
        freeze()
        return 0
    if args.cmd == 'tamper':
        return 0 if gate_tamper() else 1
    if args.cmd == 'state':
        return 0 if gate_state() else 1
    results = {'determinism': gate_determinism(),
               'avalanche': gate_avalanche(args.only),
               'diffusion': gate_diffusion(args.only),
               'tamper': gate_tamper(),
               'state': gate_state()}
    print('\nfamily gates (%s): ' % Path(args.mod).name + ', '.join(
        '%s=%s' % (k, 'PASS' if v else 'FAIL')
        for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
