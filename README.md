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

## Where it stands (after the first expedition)

The full build order of IGNITION §4 is executed and recorded (89
hash-chained records). **Read [`LEARNINGS.md`](LEARNINGS.md) for
everything measured** — the five complete breaks, the structural
findings, the design rules, and the methodology lessons. Short
version:

- **TSH-512 is broken** — free padding collisions
  (`TSH(X) == TSH(X||0x80)` for `len(X) ≡ 30 mod 32`), sealed into
  the frozen fixture.
- The **Trit Hash Family contest ran to completion**: T1 (sponge) and
  T2 (ARX) were each broken structurally, redesigned, and re-gated;
  T3 (Feistel) was dropped by its pre-registered rule; **T4-v2
  (TRIT-MD) is the family standard**, with T1-v2 / T2-v3 as
  published alternates.
- The survivors hold five pre-registered gates, 40/40 cross-language
  truth (Python ↔ Rust), and budget-labeled attack records —
  published for outside attack in **[`publish/`](publish/)**.

## Layout

| Path | What it is |
|---|---|
| `LEARNINGS.md` | **Everything the first expedition measured and learned** |
| `publish/` | Phase-5 publication bundle for OUTSIDE attack (manifest-sealed) |
| `SPEC.md` | The Trit Hash Family spec (T1–T4, gates, contest, succession) |
| `IGNITION.md` | The law of this timeline: build order, fitness function, agent protocol |
| `timeline.jsonl` | Hash-chained append-only history of this timeline |
| `proto/ternary_hash.py` | Python prototype — TSH + PDH, repaired instruments |
| `proto/ternary.py` | Port of the tryte-vm's balanced-ternary core (VM tests 44/44, 32/32 vs JS) |
| `proto/trit_family_v2.py` | The family reference: T1-v2 / T2-v3 / T3-v3(field) / T4-v2 |
| `proto/trit_family.py` | v1 family (kept as the artifact the court records describe) |
| `rust/tsh.rs`, `rust/pdh.rs` | Bit-exact std-only Rust ports of TSH-512 / PDH-512 |
| `rust/trit_family_v2.rs` | Rust core of the family (40/40 vs Python on frozen vectors) |
| `rust/verify_vectors.rs`, `rust/verify_family.rs` | Cross-language truth harnesses (line protocol) |
| `vectors/` | Frozen fixtures: TSH/PDH (incl. the known-collision pair) + family v2 |
| `tools/` | Timeline, gates (incl. the state-level gate), freeze/tamper/verify, attack instruments |
| `ref/ternary.js` | tryte-vm's audited balanced-ternary arithmetic reference |

## The timeline

Every act on this timeline is recorded in `timeline.jsonl` —
append-only, hash-chained (canonical JSON, sha256, each record names
its predecessor). `python tools/timeline.py verify` recomputes the
chain. The first record welds this timeline to its parent chain in the
Aethor chronicle.

**Quick start:**

    python tools/timeline.py verify                 # the chain (89 records)
    python tools/family_gates.py gates --mod proto/trit_family_v2.py \
        --vec vectors/trit_family_vectors_v2.json   # the five family gates
    python tools/verify_family_rust.py              # 40/40 cross-language truth
    python proto/ternary_hash.py test               # TSH/PDH instruments
