//! TSH-512: Ternary Sponge Hash.
//!
//! Bit-exact Rust port of the TSH construction in `tools/ternary_hash.py`
//! (construction 1). The sponge permutation alternates ternary gate
//! rotations (SEARCH/REFUTE/HIDE/REMEMBER quarter-turns) with Feistel
//! pairwise mixing and a column-parity step, 24 rounds per permutation —
//! the four gates are non-commutative, which is where the "ternary"
//! mixing (and the quantum-resistance hypothesis) comes from.
//!
//! Layout:
//! - State: 8 × 64-bit words (512 bits total)
//! - Rate: 4 words (256 bits absorbed per 32-byte block)
//! - Capacity: 4 words (256 bits of hidden state)
//! - Output: 512 bits (two 32-byte squeezes)
//!
//! RESEARCH construction — see the Python prototype's disclaimer: not a
//! drop-in replacement for blake2b/sha3 without peer review.

/// Quarter-turn phase constants for the four ternary gates
/// (e^(iπ/4) scaled into the integer domain).
pub const PHASE_CONSTANTS: [u64; 4] = [
    0x9E3779B97F4A7C15, // SEARCH:   (1+i)/√2  → golden ratio low
    0xC2B2AE3D27D4EB4F, // REFUTE:   (1-i)/√2  → golden ratio high
    0x165667B19E3779F9, // HIDE:     (-1+i)/√2 → inverse golden
    0x85EBCA77C2B2AE31, // REMEMBER: (-1-i)/√2 → murmur constant
];

/// Non-commutative round constants (each round shifts gate selection).
/// These are the first 24 digits of pi and sqrt(2)-style expansions —
/// ported EXACTLY from the Python; cryptographically significant.
pub const ROUND_CONSTANTS: [u64; 24] = [
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

/// Non-linear mix multipliers (SplitMix64 finalizer constants).
const MIX_MUL1: u64 = 0xBF58476D1CE4E5B9;
const MIX_MUL2: u64 = 0x94D049BB133111EB;

/// Number of 64-bit words in the state.
const STATE_WORDS: usize = 8;
/// Rate words absorbed per block (256 bits).
const RATE_WORDS: usize = 4;
/// Block size in bytes (4 words × 8 bytes).
const BLOCK_BYTES: usize = RATE_WORDS * 8;
/// Rounds per permutation (SHA-3 also uses 24).
pub const ROUNDS: usize = 24;

/// Rotate a 64-bit word left by `n` (mod 64), matching the Python `rotl64`.
#[inline]
pub fn rotl64(x: u64, n: u32) -> u64 {
    x.rotate_left(n & 63)
}

/// Apply a ternary gate to a 64-bit word: quarter-turn + XOR + multiply.
///
/// Each gate is a rotation combined with a non-linear mix. The rotation
/// amount depends on the gate, so applying gates in different orders
/// gives different results — NON-COMMUTATIVE.
///
/// - Gate 0 SEARCH:   rotate 16
/// - Gate 1 REFUTE:   rotate 48
/// - Gate 2 HIDE:     rotate 32
/// - Gate 3 REMEMBER: rotate 56
#[inline]
pub fn gate_apply(word: u64, gate_index: usize) -> u64 {
    const ROTATIONS: [u32; 4] = [16, 48, 32, 56]; // quarter-turns
    let idx = gate_index & 3;
    let mut w = rotl64(word, ROTATIONS[idx]);
    w ^= PHASE_CONSTANTS[idx];
    // non-linear mix: multiplication (like MurmurHash3's fmix)
    w ^= w >> 30;
    w = w.wrapping_mul(MIX_MUL1);
    w ^= w >> 27;
    w = w.wrapping_mul(MIX_MUL2);
    w ^= w >> 31;
    w
}

/// One permutation round on the 8×64-bit state.
///
/// Gate rotations with data-dependent gate selection, then Feistel-like
/// pairwise mixing, then a column-parity step (simplified SHA-3 theta).
/// The loop order is load-bearing: the Python prototype mutates the
/// state in place, sequentially, so each Feistel step reads words the
/// previous steps already wrote.
pub fn permute_round(state: &mut [u64; STATE_WORDS], round_num: usize) {
    let rc = ROUND_CONSTANTS[round_num % ROUND_CONSTANTS.len()];

    // Step 1: apply gates — the gate depends on round AND state content.
    for i in 0..STATE_WORDS {
        let gate_idx = round_num.wrapping_add((state[i] & 3) as usize) & 3;
        state[i] = gate_apply(state[i], gate_idx);
    }

    // Step 2: pairwise Feistel mixing (sequential, in place).
    // Python: f = rotl64(state[i] + rc, 7); state[j] ^= f  with j = (i+1) & 7.
    // The sum may carry past bit 63; Python's big-int rotl64 then ORs the
    // carry bit into bit 7 of the result. Reproduce that exactly.
    for i in 0..STATE_WORDS {
        let j = (i + 1) & 7;
        let (sum, carry) = state[i].overflowing_add(rc);
        let f = sum.rotate_left(7) | ((carry as u64) << 7);
        state[j] ^= f;
    }

    // Step 3: column mix — XOR a rotated parity of all words into each word.
    let mut parity: u64 = 0;
    for i in 0..STATE_WORDS {
        parity ^= state[i];
    }
    for i in 0..STATE_WORDS {
        state[i] ^= rotl64(parity, ((i as u32 + 1) * 8) & 63);
    }
}

/// Full permutation: `rounds` rounds of [`permute_round`] (24 by default).
pub fn permute(state: &mut [u64; STATE_WORDS], rounds: usize) {
    for r in 0..rounds {
        permute_round(state, r);
    }
}

/// 10*1 padding (like SHA-3): append 0x80, zero-fill to a 32-byte
/// multiple, then set the final byte to 0x01.
fn pad(data: &[u8]) -> Vec<u8> {
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % BLOCK_BYTES != 0 {
        padded.push(0x00);
    }
    let last = padded.len() - 1;
    padded[last] = 0x01;
    padded
}

/// Sponge state for TSH-512: 8 × 64-bit words, zero-initialized.
pub struct TshState {
    /// The 512-bit sponge state (rate = words 0..4, capacity = words 4..8).
    pub state: [u64; STATE_WORDS],
}

impl TshState {
    /// Fresh all-zero sponge state.
    pub fn new() -> Self {
        TshState {
            state: [0u64; STATE_WORDS],
        }
    }

    /// XOR one 32-byte block into the rate words, then permute.
    pub fn absorb_block(&mut self, block: &[u8; BLOCK_BYTES]) {
        for i in 0..RATE_WORDS {
            let mut word_bytes = [0u8; 8];
            word_bytes.copy_from_slice(&block[i * 8..(i + 1) * 8]);
            self.state[i] ^= u64::from_le_bytes(word_bytes);
        }
        permute(&mut self.state, ROUNDS);
    }

    /// Pad `data` and absorb it block by block.
    pub fn absorb(&mut self, data: &[u8]) {
        let padded = pad(data);
        for chunk in padded.chunks_exact(BLOCK_BYTES) {
            let mut block = [0u8; BLOCK_BYTES];
            block.copy_from_slice(chunk);
            self.absorb_block(&block);
        }
    }

    /// Squeeze output bytes from the rate words, permuting between blocks.
    pub fn squeeze(&mut self, out: &mut [u8]) {
        let mut pos = 0;
        while pos < out.len() {
            for i in 0..RATE_WORDS {
                let word_bytes = self.state[i].to_le_bytes();
                let end = (pos + 8).min(out.len());
                out[pos..end].copy_from_slice(&word_bytes[..end - pos]);
                pos += 8;
            }
            if pos < out.len() {
                permute(&mut self.state, ROUNDS);
            }
        }
    }
}

impl Default for TshState {
    fn default() -> Self {
        Self::new()
    }
}

/// TSH-512: hash `data` to a 512-bit digest (absorb → permute → squeeze).
///
/// Bit-exact port of `tsh_hash(data, output_bits=512)` from
/// `tools/ternary_hash.py`.
pub fn tsh_512(data: &[u8]) -> [u8; 64] {
    let mut sponge = TshState::new();
    sponge.absorb(data);
    let mut out = [0u8; 64];
    sponge.squeeze(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotl64_matches_semantics() {
        assert_eq!(rotl64(1, 0), 1);
        assert_eq!(rotl64(1, 64), 1); // 64 & 63 == 0, like the Python
        assert_eq!(rotl64(1, 63), 1 << 63);
        assert_eq!(rotl64(u64::MAX, 32), u64::MAX);
        assert_eq!(rotl64(0x0123456789ABCDEF, 8), 0x23456789ABCDEF01);
    }

    #[test]
    fn gates_are_non_commutative() {
        let w = 0x0F0F0F0F0F0F0F0F;
        let a_then_b = gate_apply(gate_apply(w, 0), 1);
        let b_then_a = gate_apply(gate_apply(w, 1), 0);
        assert_ne!(a_then_b, b_then_a);
    }

    #[test]
    fn permute_diffuses() {
        let mut a = [0x1111111111111111u64; 8];
        let mut b = [0x2222222222222222u64; 8];
        permute(&mut a, 2);
        permute(&mut b, 2);
        assert_ne!(a[0], b[0]);
        assert_ne!(a[7], b[7]);
    }

    #[test]
    fn padding_edge_cases() {
        // 31 bytes: single 0x01 pad byte (0x80 slot gets overwritten)
        let p31 = pad(&[0xAB; 31]);
        assert_eq!(p31.len(), 32);
        assert_eq!(*p31.last().unwrap(), 0x01);
        assert_eq!(p31[30], 0xAB);
        // 32 bytes: full extra block of padding
        let p32 = pad(&[0xAB; 32]);
        assert_eq!(p32.len(), 64);
        assert_eq!(p32[32], 0x80);
        assert_eq!(*p32.last().unwrap(), 0x01);
    }
}
