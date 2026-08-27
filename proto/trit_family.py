#!/usr/bin/env python3
"""trit_family.py — the T1–T4 Trit Hash Family (SPEC.md v0, IGNITION
PHASE 2), built ONLY on proto/ternary.py (the verified port of the
tryte-vm's audited arithmetic). std-only; no crypto, no clock, no I/O.

Predicted weaknesses for every construction were pre-registered in the
timeline (seq 26–29) BEFORE this module was written or measured.

Instantiation decisions (recorded, falsifiable):
  - Common: bytes -> trytes one-per-byte via wrap(b) (byte <= 255 <
    364, always in range). Output = 81 trytes (486 trits). Tick and
    domain string are mixed into padding/constants (SPEC doctrine 3).
  - Common padding (the TSH seq-13 lesson, hardened): message || [+13]
    || 0* || [len_hi, len_lo, tick, -13]. Length+tick sit at a FIXED
    offset from the end, so two equal padded strings force equal
    lengths, equal ticks, then equal messages - injectivity by
    construction, no marker ever overwritten. Message domain <=
    531,441 trytes; tick < 729 (T2-W2/T3-W2/T4-W1 test these bounds).
  - T1 TRIT-SPONGE: state 27 trytes (3 rows x 9 lanes), rate 9 /
    capacity 18, 18 rounds of theta (GF(3) matrix [[0,1,1],[1,0,1],
    [1,1,1]], det = 1, per trit position per 3-lane column) / rho
    (per-lane trit rotation l % 6) / pi (lane -> (7l + 3) mod 27) /
    chi (the ONLY non-linearity: x -> x^5 over GF(27), irreducible
    a^3 + 2a + 1, bijective since gcd(5, 26) = 1, applied to both
    3-trit halves of each lane) / iota (tick+domain-seeded additive
    chain).
  - T2 TRIT-ARX: BLAKE2b-shaped, 32 tryte words (two 16-word pipes),
    10 rounds of 8 G-steps each; G uses add / sub / trit-rotation /
    mul by 2 and 4 (bijections mod 729; mul by 3 is NOT and is never
    used); message words drive one pass, tick-salt words the other;
    inits = TSH prototype ROUND_CONSTANTS wrapped to trytes.
  - T3 TRIT-FEISTEL: 16 rounds over two 27-tryte halves; round
    function = keyed quadratic polynomial per position evaluated in
    the Z/729 RING (the VM's multiply is wrap(a*b) mod 729 - SPEC's
    "GF(3^6)" is not what the VM's tables compute; interpretation
    recorded at timeline seq 28); round keys from an affine
    counter-mode chain seeded by tick+domain.
  - T4 TRIT-MD: Merkle-Damgard, 16-tryte state, 16-tryte blocks,
    compression = 3 passes of pair mixing (add/sub/rot/mul-by-2); the
    final block's tick+length padding IS the domain separation.
"""
import ternary as T

TRYTE_STATES = T.TRYTE_STATES          # 729
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
    """Common injective padding: M || [+13] || 0* || [len_hi, len_lo,
    tick, -13]. Fixed end-offsets for length/tick/marker make the map
    (message, tick) -> padded injective."""
    n = len(msg_trytes)
    pad = list(msg_trytes) + [13]
    while (len(pad) + 4) % rate != 0:
        pad.append(0)
    pad += [T.wrap((n // TRYTE_STATES) % TRYTE_STATES),
            T.wrap(n % TRYTE_STATES),
            T.wrap(tick), -13]
    return pad


def output_string(trytes):
    return ''.join(T.trit_string(v) for v in trytes)


def _expand_to_81(words, seed_const):
    """Expand a small final state to the 81-tryte output (chain-mixed;
    construction-specific seed keeps expanders decoupled). T4-W3
    pre-registers that this squeeze has measurable structure."""
    out = []
    c = T.wrap(words[0] + seed_const)
    n = len(words)
    for j in range(OUT_TRYTES):
        c = T.wrap(c * 7 + words[j % n] + seed_const)
        out.append(c)
    return out


# ── GF(27) for T1's chi ─────────────────────────────────────────────
# Element = 3 balanced trits [e0, e1, e2] <-> e0 + e1*a + e2*a^2 over
# GF(3) with irreducible a^3 + 2a + 1 = 0, i.e. a^3 = a + 2 (mod 3).

def _bal(r):
    m = r % 3
    return -1 if m == 2 else m


def gf27_mul(x, y):
    c = [0] * 5
    for i in range(3):
        for j in range(3):
            c[i + j] += x[i] * y[j]
    # reduce with a^3 = a + 2, a^4 = a^2 + 2a
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


# ── T1 — TRIT-SPONGE ────────────────────────────────────────────────

_T1_RATE = 9      # trytes
_T1_ROUNDS = 18

# theta matrix over GF(3): det = 1 (invertible)
_T1_THETA = ((0, 1, 1), (1, 0, 1), (1, 1, 1))


def _t1_theta(state):
    # 3 rows x 9 lanes; mix each column's 3 trytes per trit position
    out = list(state)
    for col in range(9):
        cols = [T.to_trits(state[r * 9 + col]) for r in range(3)]
        for k in range(6):
            t = [cols[r][k] for r in range(3)]
            for r in range(3):
                s = (_T1_THETA[r][0] * t[0] + _T1_THETA[r][1] * t[1]
                     + _T1_THETA[r][2] * t[2])
                cols[r][k] = _bal(s)
        for r in range(3):
            out[r * 9 + col] = T.from_trits(cols[r])
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
    """TRIT-SPONGE: absorb (rate 9 trytes), 18-round permutation,
    squeeze 81 trytes."""
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


# ── T2 — TRIT-ARX ───────────────────────────────────────────────────

_T2_ROUNDS = 10
_T2_BLOCK = 16
_T2_SIGMA = [  # word schedule per round (BLAKE2b-shaped), 8 steps
    (0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6), (1, 0, 11, 5, 12, 2, 7, 3),
    (11, 4, 12, 2, 8, 13, 6, 15), (5, 10, 14, 0, 3, 9, 7, 1),
    (6, 14, 11, 3, 15, 5, 10, 2), (2, 12, 4, 9, 1, 7, 13, 0),
    (3, 8, 15, 6, 10, 4, 0, 11), (13, 5, 1, 14, 7, 12, 9, 2),
]


def _t2_g(v, a, b, c, d, x, y):
    """One G-step on words at indices a..d: add/sub/rotate/mul (2 and 4
    only - bijections mod 729)."""
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
    idx = _T2_SIGMA[rnd % len(_T2_SIGMA)]
    _t2_g(h, 0, 4, 8, 12, words[idx[0]], aux[idx[1]])
    _t2_g(h, 1, 5, 9, 13, words[idx[2]], aux[idx[3]])
    _t2_g(h, 2, 6, 10, 14, words[idx[4]], aux[idx[5]])
    _t2_g(h, 3, 7, 11, 15, words[idx[6]], aux[idx[7]])
    _t2_g(h, 0, 5, 10, 15, words[idx[1]], aux[idx[0]])
    _t2_g(h, 1, 6, 11, 12, words[idx[3]], aux[idx[2]])
    _t2_g(h, 2, 7, 8, 13, words[idx[5]], aux[idx[4]])
    _t2_g(h, 3, 4, 9, 14, words[idx[7]], aux[idx[6]])
    return h


def t2_hash(data, tick=0, domain='tsh512/t2'):
    """TRIT-ARX: BLAKE-shaped double pipe over 32 tryte words; the tick
    rides the salt. Block = 16 trytes."""
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
            _t2_round(h, block, s, r)   # message pass
            _t2_round(h, s, block, r)   # salt pass (double pipe)
        for i in range(16):
            h[i] = T.add(T.sub(h[i], h[16 + (i * 5) % 16]), block[i])
    return _expand_to_81(h[:16], 17)


# ── T3 — TRIT-FEISTEL ───────────────────────────────────────────────

_T3_ROUNDS = 16
_T3_HALF = 27


def t3_hash(data, tick=0, domain='tsh512/t3'):
    """TRIT-FEISTEL: 16 rounds over two 27-tryte halves; round function
    is a keyed quadratic polynomial per position in the Z/729 ring."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    dh = domain_hash(domain)
    m = pad_trytes(bytes_to_trytes(data), tick, 2)  # any rate; halved
    # fold into the two halves (add-chaining, position-mixed)
    L = [0] * _T3_HALF
    R = [0] * _T3_HALF
    for i, v in enumerate(m):
        R[i % _T3_HALF] = T.add(R[i % _T3_HALF], v)
        if i % _T3_HALF == _T3_HALF - 1:
            L, R = R, L
    # round keys: affine counter-mode chain mod 729 (T3-W2 tests this)
    k = T.wrap(tick * 101 + dh * 7 + 5)
    keys = []
    for r in range(_T3_ROUNDS):
        k = T.wrap(k * 37 + r * 11 + 3)
        keys.append(k)
    for r in range(_T3_ROUNDS):
        kr = keys[r]
        f = []
        for i in range(_T3_HALF):
            x = R[i]
            a2 = T.add(kr, i)
            a1 = T.wrap(kr * 5 + 7)
            a0 = T.wrap(kr * 11 + i * 3 + 1)
            f.append(T.wrap(a2 * x * x + a1 * x + a0
                            + R[(i + 2) % _T3_HALF]))
        newR = [T.add(L[i], f[i]) for i in range(_T3_HALF)]
        L = R
        R = newR
    return _expand_to_81(L + R, 29)


# ── T4 — TRIT-MD ────────────────────────────────────────────────────

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
    """TRIT-MD: Merkle-Damgard; the final block's tick+length padding
    IS the domain separation."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    dh = domain_hash(domain)
    m = pad_trytes(bytes_to_trytes(data), tick, _T4_STATE)
    state = [T.add(IV16[i], T.wrap(dh * (i + 1) + tick)) for i in range(16)]
    for blk in range(0, len(m), _T4_STATE):
        state = _t4_compress(state, m[blk:blk + _T4_STATE])
    return _expand_to_81(state, 43)
