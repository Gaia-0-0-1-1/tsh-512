# THE TRIT HASH FAMILY — custom ternary hashing protocols
## Spec v0 — many prototypes, adversarial selection

*Balanced-ternary permutations from the tryte-vm's arithmetic, the tick
as domain separation, commitment as the one primitive. Value forms ONLY
if the designs survive attack courts — Curl's lesson is founding law.*

---

## 1. DOCTRINE

1. OBSCURITY IS NOT SECURITY. The frontier's value is real only after
   adversarial cryptanalysis fails. IOTA's Curl (ternary, 2017) was
   broken by outside cryptanalysts; Troika was built in response. We
   pre-register: every claim of resistance is a CLAIM under the
   constitution's Article V — attackable, and worthless until attacked.
2. BALANCED TERNARY IS THE ALPHABET: trits {−1, 0, +1}; trytes (6
   trits, 729 values) as in tryte-vm; all arithmetic from its audited
   gates (src/ternary.js is the reference; ports must match its
   behavior test-for-test).
3. THE TICK IS THE DOMAIN TAG: every hash call binds a domain string
   and, where a ledger is hashed, the tick index — encoded in balanced
   ternary, mixed into the padding/round constants. Same input at two
   ticks yields two hashes (replay-hostility by construction).
4. CANONICAL TERNARY SERIALIZATION: the trio canon law extended —
   one serialization, refuse-unfaithful, cross-language fixtures (Rust
   core, JS reference, Python courts).
5. POST-QUANTUM, STATED PRECISELY: hashes are Grover-halved, not
   broken — a sound hash with a 243-trit (≈ 115.7-bit-equivalent...
   choose 486-trit ≈ 231-bit) output is PQ-adequate BY STRUCTURE; the
   PQ safety of the NATION then rests on hash-based primitives
   (Merkle/tick ranges, one-time signatures) built atop it. Ternary is
   orthogonal to PQ — the value thesis is a CUSTOM, analyzed,
   tick-native family nobody else depends on.
6. NO EXTERNAL CRYPTO — prototypes are std-only Rust at the
   chains/sovereign.rs seam; the only imports are the machine's own
   ternary arithmetic.

## 2. THE FOUR CONSTRUCTIONS (all prototyped concurrently)

T1 — TRIT-SPONGE. Keccak-shaped: a balanced-ternary state array
(9x9x3 trits or 27 lanes of 6-trit trytes); rounds of theta (column
mixing over GF(3) matrix mult — MDS-aimed), rho (trit rotations per
lane), pi (lane permutation), chi (the only non-linearity: a ternary
S-box on 3-trit columns; candidate s(x)=x^2 over GF(27) or the
tryte-vm's gate algebra), iota (round constants seeded by the
tick-domain encoding). Rate/capacity split; absorb message trytes,
squeeze 486 trits.

T2 — TRIT-ARX. Add-rotate-mix over tryte words: balanced addition
mod 3 with carry chains across trit lanes (tryte-vm's audited adder),
rotations by trit positions, mixing via balanced subtraction and
multiplication by small constants in GF(3)^k. A BLAKE-shaped double
pipe over 32 tryte words; the tick rides the salt.

T3 — TRIT-FEISTEL. A 16-round Feistel over two 27-tryte halves; the
round function is a keyed ternary polynomial evaluation over GF(3^6)
(treat each tryte as a GF(3^6) element via the VM's multiply tables);
round keys derived by a counter-mode chain seeded with the domain+tick.

T4 — TRIT-MD. Merkle-Damgård over a ternary compression function
(two-tryte-word ARX-lite mixer) with MD-strengthening: the tick and
message length encoded as final-block balanced-ternary padding — the
padding IS the domain separation.

## 3. GATES EVERY PROTOTYPE MUST PASS (pre-registered)

- DETERMINISM: byte/trit-exact fixed vectors (empty input, one trit,
  the tick=0/1/2 encodings, 729-tryte input).
- AVALANCHE: flip any single input trit → every output trit changes
  with probability in [0.6, 0.73] over >= 10,000 samples (balanced
  ternary's neutral point is 2/3, not 1/2).
- DIFFUSION (full): any input trit's influence reaches ALL output
  positions within the claimed rounds (measured, not argued).
- NO STRUCTURAL FIXED POINTS in the top rounds detectable by the
  reduced-round differential tests (the cryptanalysis court's opening
  move).
- SELF-TEST INTEGRITY: tamper any constant → vectors fail.

## 4. THE CONTEST

Four courts build concurrently; each returns a complete std-Rust
module + vectors + its OWN predicted weaknesses (the tryte-vm METHOD:
register predictions before measuring). Then the CRYPTANALYSIS court
attacks all four on reduced rounds: differential patterns, slide
properties, multicollision attempts on short outputs, padding
confusion between domains/ticks. Survivors merge toward ONE family;
the family then faces the external standard: published vectors, the
construction published for outside attack — because that is what
Curl never survived and what our value claim requires.

## 5. SUCCESSION

The survivor becomes the tick ledger's second hash option (sha256
stays until the family survives attack), the chronicle's optional
domain-separated digest, and the root of the nation's PQ signature
stack (tick-keyed Merkle signatures — a future spec).
