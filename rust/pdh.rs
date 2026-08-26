//! PDH-512: Path-DAG Hash — Rust port of `pdh_hash` in `tools/ternary_hash.py`.
//!
//! PDH has no separate compression function: the input IS a traversal.
//! Each byte selects a ternary gate (bits 0-1) and a rotation (bits 2-7),
//! walking a path through a virtual DAG of up to `depth` levels; the
//! accumulated 8x64-bit state IS the hash. Collision resistance comes from
//! the DAG's branching factor x depth, and the construction is native to
//! the DAG+Index paradigm: the hash IS the index, a byproduct of traversal.
//!
//! Byte-exact with the Python reference (see the known-answer tests in
//! `tests/hash_pdh_tests.rs`). Output packs the 8 state words
//! **little-endian**, matching `struct.pack('<Q', w)` in the reference.
//!
//! Reference: tools/ternary_hash.py (pdh_hash, gate algebra, permute).

/// Quarter-turn constants (derived from e^(i*pi/4) scaled to integer domain).
///
/// The four ternary gates as integer operations on 64-bit words; each is a
/// "quarter-turn" — a rotation combined with a non-linear mix.
const PHASE_CONSTANTS: [u64; 4] = [
    0x9E3779B97F4A7C15, // SEARCH:   (1+i)/sqrt(2) -> golden ratio low
    0xC2B2AE3D27D4EB4F, // REFUTE:   (1-i)/sqrt(2) -> golden ratio high
    0x165667B19E3779F9, // HIDE:     (-1+i)/sqrt(2) -> inverse golden
    0x85EBCA77C2B2AE31, // REMEMBER: (-1-i)/sqrt(2) -> murmur constant
];

/// Golden ratio used to absorb each input byte into the state.
const GOLDEN_RATIO: u64 = 0x9E37_79B9_7F4A_7C15;

/// Non-commutative round constants (each round shifts the gate selection).
const ROUND_CONSTANTS: [u64; 24] = [
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
    0xBE5466CF34E90C6C,
    0xC0AC29B7C97C50DD,
    0x3F84D5B5B5470917,
    0x9216D5D98979FB1B,
    0xD1310BA698DFB5AC,
    0x2FFD72DBD01ADFB7,
    0xB8E1AFED6A267E96,
    0xBA7C9045F12C7F99,
    0x24A19947B3916CF7,
    0x0801F2E2858EFC16,
    0x636920D871574E69,
    0xA458FEA3F4933D7E,
    0x0D95748F728EB658,
    0xB8CD5CF3C6A9CA23,
    0x9C3B296E52A38B27,
    0x72F1D887D6E22B3C,
    0x6D0E2E1C0B4F0A72,
    0x99E3B4C5E7A5B6D8,
    0x6F7C8A9E1D2B3C4F,
];

/// Rotate a 64-bit word left by `n` bits.
///
/// `u64::rotate_left` already reduces `n` modulo 64 (so `rotl64(x, 64) == x`),
/// which matches the Python reference's `n &= 63` semantics, including the
/// `n == 0` identity case.
#[inline]
fn rotl64(x: u64, n: u32) -> u64 {
    x.rotate_left(n)
}

/// Apply a ternary gate to a 64-bit word.
///
/// Each gate is a quarter-turn: rotate + XOR phase constant + multiply.
/// The rotation amount depends on the gate; the multiplications (the
/// MurmurHash3/splitmix finalizer constants) provide non-linear mixing
/// (avalanche). Different rotation amounts for the same input in different
/// gate order give different results: NON-COMMUTATIVE.
///
/// * Gate 0 SEARCH:   rotate 16 + XOR PHASE[0] + multiply
/// * Gate 1 REFUTE:   rotate 48 + XOR PHASE[1] + multiply
/// * Gate 2 HIDE:     rotate 32 + XOR PHASE[2] + multiply
/// * Gate 3 REMEMBER: rotate 56 + XOR PHASE[3] + multiply
#[inline]
fn gate_apply(word: u64, gate_index: usize) -> u64 {
    const ROTATIONS: [u32; 4] = [16, 48, 32, 56]; // quarter-turn offsets
    let gi = gate_index & 3;
    let mut w = rotl64(word, ROTATIONS[gi]);
    w ^= PHASE_CONSTANTS[gi];
    // non-linear mix: multiplication (like MurmurHash3's fmix)
    w ^= w >> 30;
    w = w.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    w ^= w >> 27;
    w = w.wrapping_mul(0x94D0_49BB_1331_11EB);
    w ^= w >> 31;
    w
}

/// One permutation round on the 8x64-bit state.
///
/// Gate rotations + Feistel-like pairwise mixing + a simplified SHA-3
/// theta-style column mix. Gate selection varies by round AND by state
/// content (data-dependent). Step 2 is sequential and in-place — iteration
/// order matters and mirrors the reference exactly.
fn permute_round(state: &mut [u64; 8], round_num: usize) {
    let rc = ROUND_CONSTANTS[round_num % ROUND_CONSTANTS.len()];

    // Step 1: apply gates (data-dependent gate selection)
    for i in 0..8 {
        let gate_idx = (round_num + (state[i] & 3) as usize) & 3;
        state[i] = gate_apply(state[i], gate_idx);
    }

    // Step 2: pairwise Feistel mixing (ensures diffusion)
    for i in 0..8 {
        let j = (i + 1) & 7;
        // Feistel: new_j = old_j ^ f(old_i + round_const)
        //
        // The reference computes rotl64(state[i] + rc, 7) on an unbounded
        // Python int: when the addition carries out of bit 63, the carry
        // re-enters through bit 7 of the rotate. Compute in u128 to
        // replicate that exactly (a plain wrapping_add would drop it).
        let s = state[i] as u128 + rc as u128;
        let f = ((s << 7) | (s >> 57)) as u64;
        state[j] ^= f;
    }

    // Step 3: column mix (like SHA-3's theta, simplified)
    let mut parity = 0u64;
    for i in 0..8 {
        parity ^= state[i];
    }
    for i in 0..8 {
        // (i + 1) * 8 reaches 64 at i == 7; rotl64 reduces mod 64 -> 0,
        // same as the Python reference.
        state[i] ^= rotl64(parity, (i as u32 + 1) * 8);
    }
}

/// Full permutation: `rounds` rounds (the sponge construction uses 24;
/// PDH's final diffusion uses 8).
fn permute(state: &mut [u64; 8], rounds: usize) {
    for r in 0..rounds {
        permute_round(state, r);
    }
}

/// Streaming state for the Path-DAG Hash.
///
/// Feed bytes with [`update`](PdhState::update) in any chunking; the
/// traversal is strictly sequential, so chunk boundaries never affect the
/// digest. [`finalize`](PdhState::finalize) applies the final 8-round
/// permutation and returns the 64-byte digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PdhState {
    state: [u64; 8],
}

impl PdhState {
    /// Fresh state with the fixed initial constants of the reference.
    pub fn new() -> Self {
        PdhState {
            state: [
                0x0123_4567_89AB_CDEF,
                0xFEDC_BA98_7654_3210,
                0xDEAD_BEEF_CAFE_BABE,
                0x0BAD_C0DE_0D15_EA5E,
                0xCAFE_BABE_DEAD_BEEF,
                0x0D15_EA5E_0BAD_C0DE,
                0x9876_5432_10FE_DCBA,
                0x89AB_CDEF_0123_4567,
            ],
        }
    }

    /// Walk `data` through the virtual DAG: each byte selects a gate (bits
    /// 0-1) and a rotation (bits 2-7), all words take the gate with a
    /// per-word variation, a cross-mix diffuses information across words,
    /// and the byte itself is absorbed into one word.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            // each byte selects a gate (0-3) and a rotation (0-63)
            let gate = (byte & 3) as usize;
            let rot = ((byte >> 2) & 63) as u32;

            // apply the gate to every word (with variation)
            for i in 0..8usize {
                self.state[i] = gate_apply(self.state[i], (gate + i) & 3);
                self.state[i] = rotl64(self.state[i], (rot + i as u32) & 63);
            }

            // cross-mix: information diffuses across words
            // (sequential and in-place — order matters)
            for i in 0..8usize {
                let j = (i + gate + 1) & 7;
                self.state[i] ^= self.state[j];
            }

            // absorb the byte into the state
            self.state[(byte & 7) as usize] ^= (byte as u64).wrapping_mul(GOLDEN_RATIO);
        }
    }

    /// Finish: permute the state for full diffusion, then pack the 8 words
    /// little-endian into the 64-byte digest (the state IS the hash).
    pub fn finalize(mut self) -> [u8; 64] {
        // final: permute to ensure full diffusion
        permute(&mut self.state, 8);

        let mut out = [0u8; 64];
        for (chunk, w) in out.chunks_exact_mut(8).zip(self.state) {
            chunk.copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

impl Default for PdhState {
    fn default() -> Self {
        Self::new()
    }
}

/// PDH-512 one-shot: hash `data` to 64 bytes.
///
/// Quantum security estimate (assuming strong diffusion): collision
/// O(2^(depth/6)) with BHT; ~85-bit at the default depth of 512.
pub fn pdh_512(data: &[u8]) -> [u8; 64] {
    let mut st = PdhState::new();
    st.update(data);
    st.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_matches_reference() {
        let st = PdhState::new();
        assert_eq!(st.state[0], 0x0123_4567_89AB_CDEF);
        assert_eq!(st.state[7], 0x89AB_CDEF_0123_4567);
    }

    #[test]
    fn streaming_matches_oneshot() {
        let data: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        for split in [0, 1, 7, 63, 64, 512, 999, 1000] {
            let mut st = PdhState::new();
            st.update(&data[..split]);
            st.update(&data[split..]);
            assert_eq!(st.finalize(), pdh_512(&data), "split at {split}");
        }
    }
}
