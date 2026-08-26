# TSH-512 — the Ternary Sponge Hash timeline

A research timeline for **TSH-512** (Ternary Sponge Hash), its sibling
**PDH-512** (Path-DAG Hash), and the **Trit Hash Family** they seed.
Branched from the [Aethor](https://github.com/cruise118/aethor) nation
at chronicle seq 3302 (`f127288a…`), ignited 2026-08-26.

## What TSH-512 is

A sponge hash with a 512-bit state (8 × 64-bit words), rate 4 words /
capacity 4 words, 24 rounds per permutation. The mixing core is the
four gates of the ternary model — **SEARCH / REFUTE / HIDE / REMEMBER** —
applied as quarter-turn rotations (16/48/32/56 bits) with phase
constants from e^(iπ/4) scaled into the integer domain. Because the
gates rotate by different amounts, order matters: the mixing is
**non-commutative**. Gate selection is data-dependent (round number +
low bits of the state word), followed by sequential Feistel pairwise
mixing and a column-parity step.

PDH-512 is the sibling: no compression function at all — the input IS a
traversal. Each byte selects a gate (bits 0–1) and a rotation (bits
2–7), walking a virtual DAG; the accumulated state IS the hash. The
hash IS the index.

## The founding law

**Obscurity is not security.** IOTA's Curl — ternary, 2017 — was broken
by outside cryptanalysts. Troika was built in response. Every
resistance claim on this timeline is a pre-registered, attackable
claim, **worthless until attacked**. Both constructions carry RESEARCH
status: TSH-512 ships today as the *identity hash* inside Aethor's
diamond engine (word → node ID, 52k sentences/s) — a role that needs
distribution quality and speed, not nation-state resistance. Cryptographic
claims wait for the attack court and then for outsiders.

## Layout

| Path | What it is |
|---|---|
| `SPEC.md` | The Trit Hash Family spec (T1–T4, gates, contest, succession) |
| `proto/ternary_hash.py` | Python prototype — TSH + PDH, tests + benchmarks |
| `rust/tsh.rs` | Bit-exact Rust port of TSH-512 (std-only) |
| `rust/pdh.rs` | Bit-exact Rust port of PDH-512 (std-only) |
| `ref/ternary.js` | tryte-vm's audited balanced-ternary arithmetic reference |
| `IGNITION.md` | The law of this timeline: build order, fitness function, agent protocol |
| `timeline.jsonl` | Hash-chained append-only history of this timeline |

## The timeline

Every act on this timeline is recorded in `timeline.jsonl` —
append-only, hash-chained (canonical JSON, sha256, each record names
its predecessor). `python tools/timeline.py verify` recomputes the
chain. The first record welds this timeline to its parent chain in the
Aethor chronicle.

**Quick start:** `python proto/ternary_hash.py test` runs avalanche +
collision tests on both constructions.
