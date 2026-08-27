//! trit_family_v2.rs — std-only Rust port of proto/trit_family_v2.py
//! (revision v2 + T2 iteration-3 wiring), the T1–T4 family per SPEC.md.
//!
//! Canon law: this port must agree with the Python reference
//! bit-for-bit (trit-for-trit) on every frozen vector in
//! vectors/trit_family_vectors_v2.json — one disagreement makes the
//! port WRONG, not "different" (IGNITION §2 law 4).
//!
//! All arithmetic is exact i64 with wrap() applied at exactly the same
//! points as the Python (every Python T.add/T.sub/T.mul is a wrap;
//! big-int products are wrapped once at the end where the Python does).
//! Non-negative assumptions are documented where floor-div vs
//! trunc-div could differ (message length, tick — both >= 0 here).

pub const TRYTE_STATES: i64 = 729;
pub const TRYTE_MIN: i64 = -364;
pub const OUT_TRYTES: usize = 81;

pub fn wrap(v: i64) -> i64 {
    (((v - TRYTE_MIN) % TRYTE_STATES + TRYTE_STATES) % TRYTE_STATES)
        + TRYTE_MIN
}

pub fn add(a: i64, b: i64) -> i64 {
    wrap(a + b)
}

pub fn sub(a: i64, b: i64) -> i64 {
    wrap(a - b)
}

pub fn mul(a: i64, b: i64) -> i64 {
    wrap(a * b)
}

/// Integer -> 6 balanced trits, least-significant first (wraps input).
pub fn to_trits(v: i64) -> [i8; 6] {
    let mut x = wrap(v);
    let mut out = [0i8; 6];
    for slot in out.iter_mut() {
        let mut r = ((x % 3) + 3) % 3;   // 0, 1, 2
        if r == 2 {
            r = -1;                      // 2 carries up, becomes -1
        }
        *slot = r as i8;
        x = (x - r) / 3;
    }
    out
}

/// Balanced trits (LSB first) -> integer.
pub fn from_trits(t: &[i8]) -> i64 {
    let mut v: i64 = 0;
    for i in (0..t.len()).rev() {
        v = v * 3 + t[i] as i64;
    }
    v
}

/// Human form, MSB first: '+' / '0' / '-'.
pub fn trit_string(v: i64) -> String {
    let t = to_trits(v);
    (0..6).rev()
        .map(|i| match t[i] {
            1 => '+',
            -1 => '-',
            _ => '0',
        })
        .collect()
}

/// Full output serialization: trytes in order, each MSB-first.
pub fn output_string(trytes: &[i64]) -> String {
    trytes.iter().map(|v| trit_string(*v)).collect()
}

/// Rotate a tryte's 6 trits left by n (no loss).
pub fn rotl_tryte(v: i64, n: usize) -> i64 {
    let n = n % 6;
    let t = to_trits(v);
    let mut r = [0i8; 6];
    for i in 0..6 {
        r[i] = t[(i + n) % 6];
    }
    from_trits(&r)
}

/// Domain string -> one tryte (matches Python domain_hash; domains are
/// ASCII so char codes equal Python ord()).
pub fn domain_hash(domain: &str) -> i64 {
    let mut h: i64 = 0;
    for (i, ch) in domain.chars().enumerate() {
        h = wrap(h + ((ch as i64) % TRYTE_STATES) * (i as i64 + 1));
    }
    wrap(h * 13 + 1)
}

pub fn bytes_to_trytes(data: &[u8]) -> Vec<i64> {
    data.iter().map(|b| wrap(*b as i64)).collect()
}

/// Minimal balanced-ternary digits of a non-negative tick, LSB first,
/// values in {-1, 0, 1} (unique representation).
fn tick_digits(tick: i64) -> Vec<i64> {
    if tick == 0 {
        return vec![0];
    }
    let mut t = tick;
    let mut digits = Vec::new();
    while t > 0 {
        let mut r = t % 3;
        if r == 2 {
            r = -1;
        }
        digits.push(r);
        t = (t - r) / 3;
    }
    digits
}

/// v4 injective padding: M || [+13] || tick_digits || [+2] || 0* ||
/// [len_hi, len_lo, -13]. The tick is an UNBOUNDED integer (v3's
/// two-tryte tick replayed at 729^2); [+2] cannot be a digit, so the
/// first [+2] after [+13] is an unambiguous terminator.
pub fn pad_trytes(msg: &[i64], tick: i64, rate: usize) -> Vec<i64> {
    let n = msg.len() as i64;
    let mut pad: Vec<i64> = msg.to_vec();
    pad.push(13);
    pad.extend(tick_digits(tick));
    pad.push(2);
    while (pad.len() + 3) % rate != 0 {
        pad.push(0);
    }
    pad.push(wrap((n / TRYTE_STATES) % TRYTE_STATES));
    pad.push(wrap(n % TRYTE_STATES));
    pad.push(-13);
    pad
}

/// Expand a small final state to the 81-tryte output (chain-mixed).
fn expand_to_81(words: &[i64], seed_const: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(OUT_TRYTES);
    let n = words.len();
    let mut c = wrap(words[0] + seed_const);
    for j in 0..OUT_TRYTES {
        c = wrap(c * 7 + words[j % n] + seed_const);
        out.push(c);
    }
    out
}

// ── GF(27) for T1's chi: a^3 + 2a + 1 = 0, i.e. a^3 = a + 2 ─────────

fn bal(r: i64) -> i8 {
    let m = ((r % 3) + 3) % 3;
    if m == 2 {
        -1
    } else {
        m as i8
    }
}

fn gf27_mul(x: &[i8; 3], y: &[i8; 3]) -> [i8; 3] {
    let mut c = [0i64; 5];
    for i in 0..3 {
        for j in 0..3 {
            c[i + j] += x[i] as i64 * y[j] as i64;
        }
    }
    // reduce with a^3 = a + 2, a^4 = a^2 + 2a
    [bal(c[0] + 2 * c[3]),
     bal(c[1] + c[3] + 2 * c[4]),
     bal(c[2] + c[4])]
}

fn gf27_pow5(x: &[i8; 3]) -> [i8; 3] {
    let x2 = gf27_mul(x, x);
    let x4 = gf27_mul(&x2, &x2);
    gf27_mul(&x4, x)
}

/// T1 chi on one tryte: both 3-trit halves x -> x^5 over GF(27).
fn chi_lane(v: i64) -> i64 {
    let t = to_trits(v);
    let lo = gf27_pow5(&[t[0], t[1], t[2]]);
    let hi = gf27_pow5(&[t[3], t[4], t[5]]);
    from_trits(&[lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]])
}

// ── T1 — TRIT-SPONGE (v2: cross-class theta) ────────────────────────

const T1_RATE: usize = 9;
const T1_ROUNDS: usize = 18;
const T1_THETA: [[i64; 3]; 3] = [[0, 1, 1], [1, 0, 1], [1, 1, 1]];

/// v2 theta: mix ADJACENT lane triples {3g, 3g+1, 3g+2} per trit
/// position (breaks the v1 mod-9 class invariance).
fn t1_theta(state: &mut [i64; 27]) {
    for g in 0..9usize {
        let lanes = [3 * g, 3 * g + 1, 3 * g + 2];
        let mut cols = [to_trits(state[lanes[0]]),
                        to_trits(state[lanes[1]]),
                        to_trits(state[lanes[2]])];
        for k in 0..6 {
            let t = [cols[0][k], cols[1][k], cols[2][k]];
            for r in 0..3 {
                let s = T1_THETA[r][0] * t[0] as i64
                    + T1_THETA[r][1] * t[1] as i64
                    + T1_THETA[r][2] * t[2] as i64;
                cols[r][k] = bal(s);
            }
        }
        for r in 0..3 {
            state[lanes[r]] = from_trits(&cols[r]);
        }
    }
}

fn t1_permute(state: &mut [i64; 27], seed: i64) {
    let mut rc = wrap(seed * 101 + 5);
    for r in 0..T1_ROUNDS {
        t1_theta(state);                                        // theta
        let mut rotated = [0i64; 27];
        for (l, v) in state.iter().enumerate() {                // rho
            rotated[l] = rotl_tryte(*v, l % 6);
        }
        let mut shuffled = [0i64; 27];
        for l in 0..27 {                                        // pi
            shuffled[(l * 7 + 3) % 27] = rotated[l];
        }
        for v in shuffled.iter_mut() {                          // chi
            *v = chi_lane(*v);
        }
        rc = wrap(rc * 31 + r as i64 * 7 + 5);                  // iota
        shuffled[0] = add(shuffled[0], rc);
        *state = shuffled;
    }
}

pub fn t1_hash(data: &[u8], tick: i64) -> Vec<i64> {
    let seed = wrap(tick * 13 + domain_hash("tsh512/t1"));
    let mut state = [0i64; 27];
    let padded = pad_trytes(&bytes_to_trytes(data), tick, T1_RATE);
    for blk in (0..padded.len()).step_by(T1_RATE) {
        for i in 0..T1_RATE {
            state[i] = add(state[i], padded[blk + i]);
        }
        t1_permute(&mut state, seed);
    }
    let mut out: Vec<i64> = Vec::with_capacity(OUT_TRYTES);
    while out.len() < OUT_TRYTES {
        out.extend_from_slice(&state[..T1_RATE]);
        if out.len() < OUT_TRYTES {
            t1_permute(&mut state, seed);
        }
    }
    out.truncate(OUT_TRYTES);
    out
}

// ── T2 — TRIT-ARX (v3: bidirectional pipes + cross-mix) ─────────────

const T2_ROUNDS: usize = 10;
const T2_BLOCK: usize = 16;

const T2_SIGMA: [[usize; 8]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6], [1, 0, 11, 5, 12, 2, 7, 3],
    [11, 4, 12, 2, 8, 13, 6, 15], [5, 10, 14, 0, 3, 9, 7, 1],
    [6, 14, 11, 3, 15, 5, 10, 2], [2, 12, 4, 9, 1, 7, 13, 0],
    [3, 8, 15, 6, 10, 4, 0, 11], [13, 5, 1, 14, 7, 12, 9, 2],
];
const T2_DIAG: [[usize; 4]; 4] = [
    [16, 21, 26, 31], [17, 22, 27, 28],
    [18, 23, 24, 29], [19, 20, 25, 30],
];

/// One G-step: add/sub/rotate/mul by 2 and 4 (bijections mod 729).
fn t2_g(h: &mut [i64; 32], a: usize, b: usize, c: usize, d: usize,
        x: i64, y: i64) {
    let (mut va, mut vb, mut vc, mut vd) = (h[a], h[b], h[c], h[d]);
    va = add(va, add(vb, x));
    vd = rotl_tryte(sub(vd, va), 3);
    vc = add(vc, vd);
    vb = rotl_tryte(sub(vb, vc), 9);
    va = mul(va, 2);
    vd = mul(vd, 4);
    vc = rotl_tryte(add(vc, va), 5);
    vb = add(vb, y);
    va = rotl_tryte(va, 7);
    vd = add(vd, vb);
    vb = mul(vb, 2);
    vc = sub(vc, vd);
    h[a] = va;
    h[b] = vb;
    h[c] = vc;
    h[d] = vd;
}

/// v3 round: columns AND diagonals on BOTH pipes (16 G-steps).
fn t2_round(h: &mut [i64; 32], words: &[i64], aux: &[i64], rnd: usize) {
    let idx = T2_SIGMA[rnd % T2_SIGMA.len()];
    // pipe A columns
    t2_g(h, 0, 4, 8, 12, words[idx[0]], aux[idx[1]]);
    t2_g(h, 1, 5, 9, 13, words[idx[2]], aux[idx[3]]);
    t2_g(h, 2, 6, 10, 14, words[idx[4]], aux[idx[5]]);
    t2_g(h, 3, 7, 11, 15, words[idx[6]], aux[idx[7]]);
    // pipe B columns (words/aux crossed)
    t2_g(h, 16, 20, 24, 28, aux[idx[1]], words[idx[0]]);
    t2_g(h, 17, 21, 25, 29, aux[idx[3]], words[idx[2]]);
    t2_g(h, 18, 22, 26, 30, aux[idx[5]], words[idx[4]]);
    t2_g(h, 19, 23, 27, 31, aux[idx[7]], words[idx[6]]);
    // pipe A diagonals
    t2_g(h, 0, 5, 10, 15, words[idx[1]], aux[idx[0]]);
    t2_g(h, 1, 6, 11, 12, words[idx[3]], aux[idx[2]]);
    t2_g(h, 2, 7, 8, 13, words[idx[5]], aux[idx[4]]);
    t2_g(h, 3, 4, 9, 14, words[idx[7]], aux[idx[6]]);
    // pipe B diagonals
    for (k, quad) in T2_DIAG.iter().enumerate() {
        t2_g(h, quad[0], quad[1], quad[2], quad[3],
             aux[idx[2 * k]], words[idx[2 * k + 1]]);
    }
}

pub fn t2_hash(data: &[u8], tick: i64) -> Vec<i64> {
    let dh = domain_hash("tsh512/t2");
    let mut h = [0i64; 32];
    for i in 0..16usize {
        h[i] = add(IV16[i], wrap(tick * 101 + dh * (i as i64 + 1)));
        h[16 + i] = add(IV16[(i * 7) % 16], wrap(tick * 37 + dh));
    }
    let m = pad_trytes(&bytes_to_trytes(data), tick, T2_BLOCK);
    let s: Vec<i64> = (0..16).map(|i| wrap(tick * 13 + dh + i as i64 * 7))
        .collect();
    for blk in (0..m.len()).step_by(T2_BLOCK) {
        let block: Vec<i64> = m[blk..blk + T2_BLOCK].to_vec();
        for r in 0..T2_ROUNDS {
            t2_round(&mut h, &block, &s, r);
            t2_round(&mut h, &s, &block, r);
        }
        for i in 0..16usize {                    // v3 cross-mix
            let (a, b) = (h[i], h[16 + i]);
            h[i] = add(a, rotl_tryte(b, 1 + i % 5));
            h[16 + i] = add(b, rotl_tryte(a, 3 + i % 4));
        }
        for i in 0..16usize {                    // feed-forward, both
            h[i] = add(sub(h[i], h[16 + (i * 5) % 16]), block[i]);
            h[16 + i] = add(sub(h[16 + i], h[(i * 3) % 16]),
                            block[(i * 7 + 3) % 16]);
        }
    }
    expand_to_81(&h[..16], 17)
}

// ── T3 — TRIT-FEISTEL (v3 iteration: GF(3^6) field arithmetic) ──────
//
// A tryte IS a degree-5 polynomial over GF(3) (trits LSB-first),
// multiplied modulo p(x) = x^6 + x^5 + x + 1 — irreducible (verified
// against every monic irreducible of degree <= 3 in the Python
// reference; the sieve count 116 = (3^6-3^3-3^2+3)/6 matches theory).

const T3_ROUNDS: usize = 16;
const T3_HALF: usize = 27;

/// c0..c5 of p(x) = x^6 + x^5 + x^4 + 1; x^6 == -(x^5 + x^4 + 1).
const GF36_RED: [i64; 6] = [-1, 0, 0, 0, -1, -1];

fn gf36_mul(x: &[i8; 6], y: &[i8; 6]) -> [i8; 6] {
    let mut c = [0i64; 11];
    for i in 0..6 {
        for j in 0..6 {
            c[i + j] += x[i] as i64 * y[j] as i64;
        }
    }
    for d in (6..11).rev() {
        let coef = c[d];
        if coef != 0 {
            for i in 0..6 {
                c[d - 6 + i] += coef * GF36_RED[i];
            }
        }
    }
    let mut out = [0i8; 6];
    for i in 0..6 {
        out[i] = bal(c[i]);
    }
    out
}

fn gf36_add(x: &[i8; 6], y: &[i8; 6]) -> [i8; 6] {
    let mut out = [0i8; 6];
    for i in 0..6 {
        out[i] = bal(x[i] as i64 + y[i] as i64);
    }
    out
}

pub fn t3_hash(data: &[u8], tick: i64) -> Vec<i64> {
    let dh = domain_hash("tsh512/t3");
    let m = pad_trytes(&bytes_to_trytes(data), tick, 2);
    let mut l = [0i64; T3_HALF];
    let mut r = [0i64; T3_HALF];
    for (i, v) in m.iter().enumerate() {
        r[i % T3_HALF] = add(r[i % T3_HALF], *v);
        if i % T3_HALF == T3_HALF - 1 {
            std::mem::swap(&mut l, &mut r);
        }
    }
    let mut k = wrap(tick * 101 + dh * 7 + 5);
    let mut keys = [0i64; T3_ROUNDS];
    for r in 0..T3_ROUNDS {
        k = wrap(k * 37 + r as i64 * 11 + 3);
        keys[r] = k;
    }
    for round in 0..T3_ROUNDS {
        let kr = keys[round];
        let a1 = to_trits(wrap(kr * 5 + 7));
        let mut f = [0i64; T3_HALF];
        for i in 0..T3_HALF {
            let x = to_trits(r[i]);
            let a2 = to_trits(add(kr, i as i64));
            let a0 = to_trits(wrap(kr * 11 + i as i64 * 3 + 1));
            let xx = gf36_mul(&x, &x);
            let mut t = gf36_mul(&a2, &xx);
            t = gf36_add(&t, &gf36_mul(&a1, &x));
            t = gf36_add(&t, &a0);
            t = gf36_add(&t, &to_trits(r[(i + 2) % T3_HALF]));
            f[i] = from_trits(&t);
        }
        let mut new_r = [0i64; T3_HALF];
        for i in 0..T3_HALF {
            new_r[i] = add(l[i], f[i]);
        }
        l = r;
        r = new_r;
    }
    let mut both: Vec<i64> = l.to_vec();
    both.extend_from_slice(&r);
    expand_to_81(&both, 29)
}

// ── T4 — TRIT-MD (tick binding only) ────────────────────────────────

const T4_STATE: usize = 16;
const T4_PASSES: usize = 3;

fn t4_compress(state: &mut [i64; 16], block: &[i64]) {
    for _ in 0..T4_PASSES {
        for i in 0..T4_STATE {
            let j = (i + 1) % T4_STATE;
            let a = add(state[i], block[i]);
            let b = add(mul(block[j], 2), state[j]);
            state[i] = rotl_tryte(add(a, b), (i % 5) + 1);
            state[j] = sub(state[j], a);
        }
    }
}

pub fn t4_hash(data: &[u8], tick: i64) -> Vec<i64> {
    let dh = domain_hash("tsh512/t4");
    let m = pad_trytes(&bytes_to_trytes(data), tick, T4_STATE);
    let mut state = [0i64; 16];
    for i in 0..16usize {
        state[i] = add(IV16[i], wrap(dh * (i as i64 + 1) + tick));
    }
    for blk in (0..m.len()).step_by(T4_STATE) {
        let block: Vec<i64> = m[blk..blk + T4_STATE].to_vec();
        let mut st = state;
        t4_compress(&mut st, &block);
        state = st;
    }
    expand_to_81(&state, 43)
}

/// TSH prototype round constants wrapped into trytes (shared IV16).
/// The literals are the mathematical values Python wraps (its ints are
/// unbounded), so the wrap is done in u64 arithmetic — an `as i64`
/// cast would reinterpret mod 2^64 and 2^64 is NOT 0 mod 729 (found by
/// the vector mismatch, canon law working as intended).
pub const IV16: [i64; 16] = [
    wrap_const(0x243F6A8885A308D3u64),
    wrap_const(0x13198A2E03707344u64),
    wrap_const(0xA4093822299F31D0u64),
    wrap_const(0x082EFA98EC4E6C89u64),
    wrap_const(0x452821E638D01377u64),
    wrap_const(0xBE5466CF34E90C6Cu64),
    wrap_const(0xC0AC29B7C97C50DDu64),
    wrap_const(0x3F84D5B5B5470917u64),
    wrap_const(0x9216D5D98979FB1Bu64),
    wrap_const(0xD1310BA698DFB5ACu64),
    wrap_const(0x2FFD72DBD01ADFB7u64),
    wrap_const(0xB8E1AFED6A267E96u64),
    wrap_const(0xBA7C9045F12C7F99u64),
    wrap_const(0x24A19947B3916CF7u64),
    wrap_const(0x0801F2E2858EFC16u64),
    wrap_const(0x636920D871574E69u64),
];

/// const-context wrap of a NON-NEGATIVE u64 (exact: v + 364 < 2^64).
const fn wrap_const(v: u64) -> i64 {
    (((v + 364) % 729) as i64) - 364
}
