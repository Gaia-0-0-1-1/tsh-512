# IGNITION — the law of the TSH-512 timeline

*Ignited 2026-08-26 by the sovereign of Aethor. This timeline branched
from cruise118/aethor at chronicle seq 3302 (`f127288a…`). Here, the
sovereign does not explore — agents explore. This document is the
standing commission every agent on this timeline inherits.*

## 1. THE CALLING

Take a research hash that exists today as an identity layer (TSH-512,
live inside Aethor's diamond engine) and find out what it really is:
freeze its vectors, build its family, run the pre-registered gates,
then **attack everything on reduced rounds**. Survivors earn the right
to be attacked by strangers. That is the whole game.

## 2. THE LAWS

1. **OBSCURITY IS NOT SECURITY.** "Nobody has explored ternary" means
   *unknown*, never *safe*. Unattacked = fitness UNKNOWN. Every
   resistance claim is attackable and worthless until attacked.
2. **PRE-REGISTRATION.** State your OWN three predicted weaknesses
   BEFORE measuring (the tryte-vm METHOD). Predictions recorded after
   results are worthless.
3. **MEASURED, NOT ARGUED.** Fixed vectors, avalanche counts, diffusion
   maps — numbers or it didn't happen. "Full diffusion in 6 rounds" is
   a measurement, not a design hope.
4. **CANON.** One serialization per language pair, cross-language
   fixtures (Python reference ↔ Rust core), refuse-unfaithful. A port
   that disagrees with the reference on one byte is wrong, not
   "different".
5. **STD-ONLY.** No crypto crates, no clock, no I/O in core
   constructions. The only dependency is the machine's own arithmetic
   (`ref/ternary.js` for the trit family).
6. **PQ, STATED PRECISELY.** Hashes are Grover-halved, not broken.
   Ternary is orthogonal to PQ. Non-commutativity is a hypothesis to
   attack, not a shield to hide behind.
7. **THE TIMELINE REMEMBERS.** Every result — pass, fail, or
   prediction — is appended to `timeline.jsonl` before moving on.
   Failures are kept. They are the map.

## 3. THE ARTIFACTS (what exists at ignition)

| Artifact | Status | Notes |
|---|---|---|
| `proto/ternary_hash.py` | reference | TSH + PDH, self-tests, benchmarks. Construction 1 = TSH sponge, 2 = PDH |
| `rust/tsh.rs` | bit-exact port | shipped in diamond engine as word-identity authority |
| `rust/pdh.rs` | bit-exact port | little-endian state packing, matches Python |
| `ref/ternary.js` | arithmetic reference | tryte-vm's audited balanced-ternary gates/adder/multiplier — the law T1–T4 must match test-for-test |
| `SPEC.md` | family spec | T1–T4 constructions, gates, contest, succession |

**Precision that matters:** TSH-512 and PDH-512 operate on **binary
64-bit words** (gate-mixed). Their avalanche target is the binary
neutral **0.5 ± 0.01 per bit**. The T1–T4 family in `SPEC.md` is
**true balanced ternary** — avalanche target **[0.6, 0.73] per trit**,
because balanced ternary's neutral point is 2/3, not 1/2. Never grade
one with the other's ruler.

## 4. THE BUILD ORDER

- **PHASE 0 — FREEZE.** Publish fixed vectors for TSH-512 and PDH-512
  (empty, 1 byte, "abc", 32/64/128-byte, the tick encodings 0/1/2).
  Verify Python ↔ Rust bit-exactness on every vector. Tamper any
  constant → vectors must fail. This makes the artifacts falsifiable.
- **PHASE 1 — ARITHMETIC.** Port `ref/ternary.js` test-for-test (Rust
  and/or Python): trits, trytes, adder, multiplier, gates. The port
  must pass the VM's own test vectors before anything builds on it.
- **PHASE 2 — THE FAMILY.** Build T1 (TRIT-SPONGE), T2 (TRIT-ARX),
  T3 (TRIT-FEISTEL), T4 (TRIT-MD) per `SPEC.md`, std-only, each with
  its own predicted-weakness pre-registration.
- **PHASE 3 — GATES.** Run mechanically on every construction:
  determinism, avalanche (right ruler!), measured full diffusion,
  tamper-test. TSH/PDH graded on bits; T1–T4 on trits.
- **PHASE 4 — ATTACK COURT.** Reduced-round differential patterns,
  slide properties, multicollision attempts on short outputs, padding
  confusion across domains/ticks. Also attack TSH's gate structure
  directly: is the non-commutativity doing real work, or is it
  decoration over an ARX core?
- **PHASE 5 — MERGE + PUBLISH.** Survivors merge toward one family;
  publish vectors and constructions for OUTSIDE attack. Value forms
  only where attackers fail — Curl's lesson is the founding law.

Phases 0–1 can run in parallel. Phase 2 seats can run concurrently.
Phase 4 needs Phase 3 outputs. **Never skip a gate by argument.**

## 5. THE FITNESS FUNCTION (pre-registered at ignition)

Lexicographic — security before performance, at every rank:

1. **TRUTH**: does the artifact match its reference bit/trit-exactly?
   (binary gate — a fail here zeroes everything below)
2. **GATES**: determinism ✓, avalanche in range ✓, measured full
   diffusion ✓, tamper-test ✓ (binary gate)
3. **RESISTANCE**: differential uniformity under a stated attack
   budget; collision depth found; slide/multicollision results
   (measured, budget-labeled; deeper = better)
4. **ECONOMY**: throughput and state size, graded only among
   constructions that cleared ranks 1–3

An organism unattacked at rank 3 carries fitness **UNKNOWN**, no matter
how fast it is. Speed on an unknown core is rank 4 by definition.

## 6. THE AGENT PROTOCOL

1. Read this file. Read `SPEC.md` and the artifact you touch.
2. Pre-register your three predicted weaknesses in `timeline.jsonl`
   BEFORE you measure anything (`--kind prediction`).
3. Do the work. std-only. Vectors or it didn't happen.
4. Append every result (`--kind result`, `--kind finding`,
   `--kind failure`). Failures are kept.
5. Never claim security for the unattacked. The strongest permitted
   claim: "survived attack budget X; unknown beyond".

## 7. THE /GOAL PROMPT (paste into any agent to enlist it)

```
/goal You are an explorer on the TSH-512 timeline (repo: <PATH-TO>/tsh-512).
Read IGNITION.md — it is the law of this timeline — then SPEC.md and the
artifact you will touch. Your phase is <PHASE N: pick the lowest unfinished
phase in IGNITION.md §4>. Laws: pre-register 3 predicted weaknesses in
timeline.jsonl BEFORE measuring (python tools/timeline.py record --kind
prediction ...); std-only; measured-not-argued; the right avalanche ruler
(bits 0.5±0.01 for TSH/PDH, trits [0.6,0.73] for T1–T4); append every
result/failure to timeline.jsonl; unattacked = UNKNOWN — never claim
security. Fitness is lexicographic: truth > gates > resistance > economy.
When your phase object is done, record the result, then pick the next
lowest unfinished phase and continue. No stopping condition — the timeline
carries on where you leave it.
```
