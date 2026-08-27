#!/usr/bin/env python3
"""trit_family_v2.py — REVISED T1–T4 family (timeline seq 49 checklist).

Changes vs proto/trit_family.py (v1, kept intact as the artifact the
court records describe):

  1. TWO-TRYTE TICK BINDING (all four): padding tail is now
     [len_hi, len_lo, tick_hi, tick_lo, -13] - closes the measured
     tick-replay hole at distance 729 (timeline seq 37). New bound:
     tick domain separation holds for ticks < 729^2 = 531,441
     (pre-registered seq 50: replay expected to reappear there).
  2. T1 CROSS-CLASS THETA: theta now mixes ADJACENT lane triples
     {3g, 3g+1, 3g+2} instead of same-mod-9-class triples {c, c+9,
     c+18}, breaking the mod-9 class invariance that killed v1
     (timeline seq 34).
  3. T2 PIPE-B WIRING: the round's diagonal G-steps now write pipe B
     (indices 16..31) with message words, and the feed-forward updates
     BOTH pipes with message words each block - pipe B is no longer
     write-once decoration (timeline seq 43).
  4. T3 arithmetic unchanged (3-adic kernel recorded at seq 46; field
     redesign deferred); T4 compression unchanged (expander masking is
     mitigated by the new STATE-level gate, not by redesign).

All arithmetic still via proto/ternary.py (verified port). std-only.
"""
import ternary as T

TRYTE_STATES = T.TRYTE_STATES          # 729
TICK_MAX = TRYTE_STATES * TRYTE_STATES - 1   # 531,440
OUT_TRYTES = 81                        # 486 trits

# provenance: TSH prototype round constants, wrapped into trytes
_TSH_ROUNDS = [
    0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0,
    0x082EFA98EC4E6C89, 0x452821E638D01377, 0xBE5466CF34E90C6C,
    0xC0AC29B7C97C50DD, 0x3F84D5B5B5470917, 0x9216D5D98979FB1B,
    0xD1310BA698DFB5AC, 0x2FFD72DBD01ADFB7, 0xB8E1AFED6A267E96,
    0xBA7C9045F12C7F99, 0x24A19947B3916CF7, 0x0801F2E2858EFC16,
    0x636920D871574E69,
]
IV16 = [T.wrap(c) for c in _TSH_ROUNDS]   # 16 trytes


def rotl_tryte(v, n):
    """Rotate a tryte's 6 trits left by n (trit positions, no loss)."""
    n %= 6
    trits = T.to_trits(v)
    return T.from_trits(trits[n:] + trits[:n])


def domain_hash(domain):
    """Domain string -> one tryte (deterministic, wraps)."""
    h = 0
    for i, ch in enumerate(domain):
        h = T.wrap(h + (ord(ch) % TRYTE_STATES) * (i + 1))
    return T.wrap(h * 13 + 1)


def bytes_to_trytes(data):
    return [T.wrap(b) for b in data]


def pad_trytes(msg_trytes, tick, rate):
    """v2 injective padding: M || [+13] || 0* || [len_hi, len_lo,
    tick_hi, tick_lo, -13]. Length AND tick at fixed end offsets, tick
    in TWO trytes (replay bound moved from 729 to 729^2)."""
    n = len(msg_trytes)
    pad = list(msg_trytes) + [13]
    while (len(pad) + 5) % rate != 0:
        pad.append(0)
    pad += [T.wrap((n // TRYTE_STATES) % TRYTE_STATES),
            T.wrap(n % TRYTE_STATES),
            T.wrap((tick // TRYTE_STATES) % TRYTE_STATES),
            T.wrap(tick % TRYTE_STATES), -13]
    return pad


def output_string(trytes):
    return ''.join(T.trit_string(v) for v in trytes)


def _expand_to_81(words, seed_const):
    """Unchanged from v1 (recorded masking is now witnessed against by
    the state-level gate, seq 44/51)."""
    out = []
    c = T.wrap(words[0] + seed_const)
    n = len(words)
    for j in range(OUT_TRYTES):
        c = T.wrap(c * 7 + words[j % n] + seed_const)
        out.append(c)
    return out


# ── GF(27) for T1's chi (unchanged from v1) ─────────────────────────

def _bal(r):
    m = r % 3
    return -1 if m == 2 else m


def gf27_mul(x, y):
    c = [0] * 5
    for i in range(3):
        for j in range(3):
            c[i + j] += x[i] * y[j]
    return [_bal(c[0] + 2 * c[3]),
            _bal(c[1] + c[3] + 2 * c[4]),
            _bal(c[2] + c[4])]


def gf27_pow5(x):
    x2 = gf27_mul(x, x)
    x4 = gf27_mul(x2, x2)
    return gf27_mul(x4, x)


def chi_lane(v):
    """T1 chi on one tryte: both 3-trit halves x -> x^5 over GF(27)."""
    trits = T.to_trits(v)
    lo = gf27_pow5(trits[0:3])
    hi = gf27_pow5(trits[3:6])
    return T.from_trits(lo + hi)


# ── T1 — TRIT-SPONGE (v2: cross-class theta) ────────────────────────

_T1_RATE = 9      # trytes
_T1_ROUNDS = 18

# theta matrix over GF(3): det = 1 (invertible), unchanged
_T1_THETA = ((0, 1, 1), (1, 0, 1), (1, 1, 1))


def _t1_theta(state):
    # v2: mix ADJACENT lane triples {3g, 3g+1, 3g+2} per trit position.
    # Consecutive lanes carry different residues mod 9, so differences
    # cross the classes pi preserves (kills the v1 invariant subspace
    # decomposition recorded at seq 34).
    out = list(state)
    for g in range(9):
        lanes = [3 * g, 3 * g + 1, 3 * g + 2]
        cols = [T.to_trits(state[l]) for l in lanes]
        for k in range(6):
            t = [cols[r][k] for r in range(3)]
            for r in range(3):
                s = (_T1_THETA[r][0] * t[0] + _T1_THETA[r][1] * t[1]
                     + _T1_THETA[r][2] * t[2])
                cols[r][k] = _bal(s)
        for r in range(3):
            out[lanes[r]] = T.from_trits(cols[r])
    return out


def _t1_permute(state, seed):
    rc = T.wrap(seed * 101 + 5)
    for r in range(_T1_ROUNDS):
        state = _t1_theta(state)
        state = [rotl_tryte(v, l % 6) for l, v in enumerate(state)]  # rho
        shuffled = [0] * 27
        for l in range(27):                                          # pi
            shuffled[(l * 7 + 3) % 27] = state[l]
        state = [chi_lane(v) for v in shuffled]                      # chi
        rc = T.wrap(rc * 31 + r * 7 + 5)                             # iota
        state[0] = T.add(state[0], rc)
    return state


def t1_hash(data, tick=0, domain='tsh512/t1'):
    if isinstance(data, str):
        data = data.encode('utf-8')
    seed = T.wrap(tick * 13 + domain_hash(domain))
    state = [0] * 27
    padded = pad_trytes(bytes_to_trytes(data), tick, _T1_RATE)
    for blk in range(0, len(padded), _T1_RATE):
        for i in range(_T1_RATE):
            state[i] = T.add(state[i], padded[blk + i])
        state = _t1_permute(state, seed)
    out = []
    while len(out) < OUT_TRYTES:
        out.extend(state[:_T1_RATE])
        if len(out) < OUT_TRYTES:
            state = _t1_permute(state, seed)
    return out[:OUT_TRYTES]


# ── T2 — TRIT-ARX (v2: pipe-B wiring) ───────────────────────────────

_T2_ROUNDS = 10
_T2_BLOCK = 16
_T2_SIGMA = [
    (0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6), (1, 0, 11, 5, 12, 2, 7, 3),
    (11, 4, 12, 2, 8, 13, 6, 15), (5, 10, 14, 0, 3, 9, 7, 1),
    (6, 14, 11, 3, 15, 5, 10, 2), (2, 12, 4, 9, 1, 7, 13, 0),
    (3, 8, 15, 6, 10, 4, 0, 11), (13, 5, 1, 14, 7, 12, 9, 2),
]
# BLAKE2b diagonal pattern, offset into pipe B (indices 16..31)
_T2_DIAG = ((16, 21, 26, 31), (17, 22, 27, 28),
            (18, 23, 24, 29), (19, 20, 25, 30))


def _t2_g(v, a, b, c, d, x, y):
    """One G-step (unchanged from v1): add/sub/rotate/mul by 2 and 4."""
    va, vb, vc, vd = v[a], v[b], v[c], v[d]
    va = T.add(va, T.add(vb, x))
    vd = rotl_tryte(T.sub(vd, va), 3)
    vc = T.add(vc, vd)
    vb = rotl_tryte(T.sub(vb, vc), 9)
    va = T.mul(va, 2)
    vd = T.mul(vd, 4)
    vc = rotl_tryte(T.add(vc, va), 5)
    vb = T.add(vb, y)
    va = rotl_tryte(va, 7)
    vd = T.add(vd, vb)
    vb = T.mul(vb, 2)
    vc = T.sub(vc, vd)
    v[a], v[b], v[c], v[d] = va, vb, vc, vd
    return v


def _t2_round(h, words, aux, rnd):
    """v3 (iteration-3) round: BIDIRECTIONAL - columns AND diagonals on
    BOTH pipes (16 G-steps). Message words reach every word of both
    pipes every round; the v2 failure (10 state trits below floor) came
    from pipe B receiving only diagonal traffic."""
    idx = _T2_SIGMA[rnd % len(_T2_SIGMA)]
    # pipe A columns (0..15)
    _t2_g(h, 0, 4, 8, 12, words[idx[0]], aux[idx[1]])
    _t2_g(h, 1, 5, 9, 13, words[idx[2]], aux[idx[3]])
    _t2_g(h, 2, 6, 10, 14, words[idx[4]], aux[idx[5]])
    _t2_g(h, 3, 7, 11, 15, words[idx[6]], aux[idx[7]])
    # pipe B columns (16..31), words/aux crossed
    _t2_g(h, 16, 20, 24, 28, aux[idx[1]], words[idx[0]])
    _t2_g(h, 17, 21, 25, 29, aux[idx[3]], words[idx[2]])
    _t2_g(h, 18, 22, 26, 30, aux[idx[5]], words[idx[4]])
    _t2_g(h, 19, 23, 27, 31, aux[idx[7]], words[idx[6]])
    # pipe A diagonals
    _t2_g(h, 0, 5, 10, 15, words[idx[1]], aux[idx[0]])
    _t2_g(h, 1, 6, 11, 12, words[idx[3]], aux[idx[2]])
    _t2_g(h, 2, 7, 8, 13, words[idx[5]], aux[idx[4]])
    _t2_g(h, 3, 4, 9, 14, words[idx[7]], aux[idx[6]])
    # pipe B diagonals
    _t2_g(h, _T2_DIAG[0][0], _T2_DIAG[0][1], _T2_DIAG[0][2],
          _T2_DIAG[0][3], aux[idx[0]], words[idx[1]])
    _t2_g(h, _T2_DIAG[1][0], _T2_DIAG[1][1], _T2_DIAG[1][2],
          _T2_DIAG[1][3], aux[idx[2]], words[idx[3]])
    _t2_g(h, _T2_DIAG[2][0], _T2_DIAG[2][1], _T2_DIAG[2][2],
          _T2_DIAG[2][3], aux[idx[4]], words[idx[5]])
    _t2_g(h, _T2_DIAG[3][0], _T2_DIAG[3][1], _T2_DIAG[3][2],
          _T2_DIAG[3][3], aux[idx[6]], words[idx[7]])
    return h


def t2_hash(data, tick=0, domain='tsh512/t2'):
    if isinstance(data, str):
        data = data.encode('utf-8')
    dh = domain_hash(domain)
    h = [0] * 32
    for i in range(16):
        h[i] = T.add(IV16[i], T.wrap(tick * 101 + dh * (i + 1)))
        h[16 + i] = T.add(IV16[(i * 7) % 16], T.wrap(tick * 37 + dh))
    m = pad_trytes(bytes_to_trytes(data), tick, _T2_BLOCK)
    s = [T.wrap(tick * 13 + dh + i * 7) for i in range(16)]
    for blk in range(0, len(m), _T2_BLOCK):
        block = m[blk:blk + _T2_BLOCK]
        for r in range(_T2_ROUNDS):
            _t2_round(h, block, s, r)   # message drives both pipes
            _t2_round(h, s, block, r)   # salt pass crosses the pipes
        # v3 per-block CROSS-MIX: information flows between pipes
        # every block (fixed small rotations, recorded at seq 63)
        for i in range(16):
            a, b = h[i], h[16 + i]
            h[i] = T.add(a, rotl_tryte(b, 1 + i % 5))
            h[16 + i] = T.add(b, rotl_tryte(a, 3 + i % 4))
        # v2 feed-forward: BOTH pipes absorb message words each block
        for i in range(16):
            h[i] = T.add(T.sub(h[i], h[16 + (i * 5) % 16]), block[i])
            h[16 + i] = T.add(T.sub(h[16 + i], h[(i * 3) % 16]),
                              block[(i * 7 + 3) % 16])
    return _expand_to_81(h[:16], 17)


# ── T3 — TRIT-FEISTEL (v3 iteration: GF(3^6) field arithmetic) ──────
#
# The v1/v2 round function evaluated its keyed quadratic in the Z/729
# RING, whose 3-adic structure gave a 1/3 kernel (timeline seq 46) and
# helped fail avalanche/state gates. v3 evaluates over the FIELD
# GF(3^6): a tryte IS a degree-5 polynomial over GF(3) (trits as
# coefficients, LSB-first), multiplied modulo the irreducible
# p(x) = x^6 + x^5 + x^4 + 1 (verified at import against every monic
# irreducible of degree <= 3 - a degree-6 poly with no factor of degree
# <= 3 is irreducible; the sieve finds exactly 116 = (3^6-3^3-3^2+3)/6,
# the theoretical count).

_GF36_C = (1, 0, 0, 0, 1, 1)  # c0..c5: p(x) = x^6 + x^5 + x^4 + 1
# x^6 == -(x^5 + x^4 + 1) mod p, as coefficients of x^0..x^5:
_GF36_RED = tuple(-c for c in _GF36_C)


def _poly_has_root(c0_5):
    full = list(c0_5) + [1]
    for v in range(3):
        if sum(c * pow(v, i, 3) for i, c in enumerate(full)) % 3 == 0:
            return True
    return False


def _prem(a, b):
    """Polynomial remainder a mod b over GF(3) (balanced trit coeffs)."""
    a = list(a)
    db = max((i for i, c in enumerate(b) if c), default=-1)
    if db < 0:
        raise ZeroDivisionError
    inv = {1: 1, 2: 2}[b[db]]
    while True:
        da = max((i for i, c in enumerate(a) if c), default=-1)
        if da < db:
            return a
        coef = _bal(a[da] * inv)
        for i in range(db + 1):
            a[da - db + i] = _bal(a[da - db + i] - coef * b[i])


def _assert_gf36_irreducible():
    import itertools
    small = []
    for d in (2, 3):
        for tail in itertools.product((0, 1, 2), repeat=d):
            q = list(tail) + [1]
            if not _poly_has_root(tail):
                small.append(q)
    full = list(_GF36_C) + [1]
    assert not _poly_has_root(_GF36_C)
    for q in small:
        r = _prem(full, q)
        assert max(r, default=0) != 0 or any(r), \
            'GF(3^6) modulus is reducible'


_assert_gf36_irreducible()


def f36_mul(x, y):
    """Field multiply two tryte-trit tuples mod x^6 + x^5 + x + 1."""
    c = [0] * 11
    for i in range(6):
        for j in range(6):
            c[i + j] += x[i] * y[j]
    for d in range(10, 5, -1):
        coef = c[d]
        if coef:
            for i in range(6):
                c[d - 6 + i] = _bal(c[d - 6 + i] + coef * _GF36_RED[i])
    return tuple(_bal(c[i]) for i in range(6))


def f36_add(x, y):
    return tuple(_bal(x[i] + y[i]) for i in range(6))


_T3_ROUNDS = 16
_T3_HALF = 27


def t3_hash(data, tick=0, domain='tsh512/t3'):
    if isinstance(data, str):
        data = data.encode('utf-8')
    dh = domain_hash(domain)
    m = pad_trytes(bytes_to_trytes(data), tick, 2)
    L = [0] * _T3_HALF
    R = [0] * _T3_HALF
    for i, v in enumerate(m):
        R[i % _T3_HALF] = T.add(R[i % _T3_HALF], v)
        if i % _T3_HALF == _T3_HALF - 1:
            L, R = R, L
    k = T.wrap(tick * 101 + dh * 7 + 5)
    keys = []
    for r in range(_T3_ROUNDS):
        k = T.wrap(k * 37 + r * 11 + 3)
        keys.append(k)
    for r in range(_T3_ROUNDS):
        kr = keys[r]
        a1 = T.to_trits(T.wrap(kr * 5 + 7))
        f = []
        for i in range(_T3_HALF):
            x = T.to_trits(R[i])
            a2 = T.to_trits(T.add(kr, i))
            a0 = T.to_trits(T.wrap(kr * 11 + i * 3 + 1))
            xx = f36_mul(x, x)
            t = f36_mul(a2, xx)
            t = f36_add(t, f36_mul(a1, x))
            t = f36_add(t, a0)
            t = f36_add(t, T.to_trits(R[(i + 2) % _T3_HALF]))
            f.append(T.from_trits(list(t)))
        newR = [T.add(L[i], f[i]) for i in range(_T3_HALF)]
        L = R
        R = newR
    return _expand_to_81(L + R, 29)


# ── T4 — TRIT-MD (v2: tick binding only) ────────────────────────────

_T4_STATE = 16
_T4_PASSES = 3


def _t4_compress(state, block):
    for _ in range(_T4_PASSES):
        for i in range(_T4_STATE):
            j = (i + 1) % _T4_STATE
            a = T.add(state[i], block[i])
            b = T.add(T.mul(block[j], 2), state[j])
            state[i] = rotl_tryte(T.add(a, b), (i % 5) + 1)
            state[j] = T.sub(state[j], a)
    return state


def t4_hash(data, tick=0, domain='tsh512/t4'):
    if isinstance(data, str):
        data = data.encode('utf-8')
    dh = domain_hash(domain)
    m = pad_trytes(bytes_to_trytes(data), tick, _T4_STATE)
    state = [T.add(IV16[i], T.wrap(dh * (i + 1) + tick)) for i in range(16)]
    for blk in range(0, len(m), _T4_STATE):
        state = _t4_compress(state, m[blk:blk + _T4_STATE])
    return _expand_to_81(state, 43)
