# THE TRIT HASH FAMILY — publication for outside attack

*Published from the TSH-512 timeline (branch of cruise118/aethor,
chronicle seq 3302). Law: IGNITION.md — obscurity is not security;
every resistance claim below is budget-labeled; unattacked = UNKNOWN.*

## What this is

A custom balanced-ternary hash family, built std-only on the tryte-vm's
audited arithmetic. After a pre-registered gate contest and internal
attack court, **T4-v2 (TRIT-MD) is the family standard**; T1-v2
(sponge) and T2-v3 (ARX) are published alternates. All three:

- pass all five pre-registered gates (determinism; avalanche in the
  trit band [0.6, 0.73]; measured diffusion; tamper; state-level
  diffusion),
- agree cross-language bit-for-bit: Python reference vs Rust core,
  40/40 frozen vectors,
- survived the internal attack budgets listed in ATTACK_LEDGER.md.

## Claims header — read before attacking

The strongest claim this timeline makes: **"survived attack budget X;
unknown beyond."** Everything here is a research artifact. It is NOT a
drop-in replacement for blake2b/sha3.

**Published bounds and known weaknesses (stated before you find them):**

1. **Tick replay at distance 729² = 531,441** (all constructions):
   `H(msg, t) == H(msg, t + 531441)` exactly. Tick domain separation
   holds only for ticks < 531,441.
2. **T4's output expander** chains `c = wrap(c*7 + w[j%16] + seed)` —
   we expect state-word leakage from digests to be the first fruitful
   outside attack (pre-registered at timeline seq 84).
3. **T4 state diffusion** has 1/96 chaining trits at 0.573 change rate
   (below the 0.6 band center, above the 0.5 floor) at full 3 passes.
4. **Economy ranking** (why T4 leads) is Python-prototype scale:
   T4 0.123 > T2 0.0032 > T1 0.0008 MB/s — instrument-labeled, native
   ratios will differ (pre-registered at seq 85).
5. **Founding artifacts TSH-512/PDH-512 are published WITH their
   breaks**: TSH has a free padding collision `TSH(X) == TSH(X||0x80)`
   for `len(X) ≡ 30 (mod 32)` (in the fixture as a known-collision
   pair); PDH has 16 dead round constants and 0x00-absorb-neutrality.
   They are the record's starting point, not sound hashes.

## Verify (std-only; no build deps beyond rustc/python/node)

    # cross-language truth (40/40):
    rustc --edition 2021 -O constructions/verify_family.rs -o verify_family
    python3 tools/../verify_family_rust.py   # or feed lines yourself:
    #   echo "t4 0 616263" | ./verify_family
    # Python reference determinism against the frozen vectors:
    python3 -c "import sys; sys.path.insert(0,'constructions'); \
        import trit_family_v2 as F, json; \
        vecs=json.load(open('vectors/trit_family_vectors_v2.json'))['vectors']; \
        print(all(F.output_string(getattr(F,{'T1':'t1_hash','T2':'t2_hash','T3':'t3_hash','T4':'t4_hash'}[t])(bytes.fromhex(r['input_hex']),r['tick']))==r[t] for r in vecs for t in ('T1','T2','T3','T4')))"

The arithmetic base (`constructions/ternary.py`) is a test-for-test
port of the tryte-vm's audited `ref/ternary.js`: 44/44 of the VM's own
test vectors, 32/32 exhaustive differential streams vs the JS.

## What counts as a break

Any pair `H(x) == H(y)` with `x != y` constructible faster than
birthday; any recovery of internal state from digests; any differential
characteristic with probability materially above random at reduced
rounds; any violation of the published bounds above. Findings belong on
the timeline (append-only, hash-chained `timeline.jsonl` is included).

*Curl's lesson is founding law: value forms only where attackers fail.
Attack this.*
