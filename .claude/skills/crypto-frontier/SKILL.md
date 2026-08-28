# Crypto Frontier Exploration

*Push custom cryptographic primitives from concept through adversarial
evaluation to publication — without a massive engineering team, using
structure, pre-registration, and instrument discipline instead of headcount.*

## The one law

**Falsifiability is the forcing function for honesty.** Every claim is a
pre-registered, attackable, budget-labeled statement. "Survived attack
budget X; unknown beyond" is the strongest sentence. "Secure" is never
written.

---

## The constitution (non-negotiable)

1. **PRE-REGISTER before measuring.** Write falsifiable predictions to
   the ledger BEFORE any data exists. Predictions recorded after results
   are worthless. If an agent dies before registering, the replacement
   inherits the binding registrations — never re-register after seeing
   partial results.

2. **CONTROLS PROVE THE INSTRUMENT.** Every measurement instrument must
   carry a control that demonstrates it can detect the thing it claims
   to measure. A collision test must detect the 16-bit-truncated
   reference. A tamper gate must detect a known-bad variant. A completion
   watcher must fire on a test condition. **A check that has only ever
   passed has demonstrated nothing.**

3. **BUDGET-LABEL EVERY CLAIM.** The strongest sentence is "survived
   attack budget N; unknown beyond." Never "secure." Never "resistant."
   Never "immune." The budget is part of the claim.

4. **FAILURES ARE THE MAP.** A falsified prediction is more informative
   than a confirmed one — it narrows the search space and forces a
   mechanism revision. Keep every failure. Corrections are new records
   naming what they correct, never edits.

5. **CROSS-LANGUAGE TRUTH IS RANK 1.** A construction that cannot be
   verified independently (different language, different implementer)
   has no truth status. Bit-exact or it doesn't count.

6. **APPEND-ONLY.** The ledger is the memory. Never edit; corrections
   are new records. The chain verifies itself.

7. **THE RIGHT RULER.** Binary avalanche: 0.5 ± 0.01. Ternary avalanche:
   [0.6, 0.73] (neutral = 2/3). Golden-mean digits: ~0.47 (derive from
   the actual measure — never reuse a ruler from a different alphabet).

---

## The gate hierarchy

Every construction passes through these gates IN ORDER. Each gate is a
binary PASS/FAIL. A gate failure at rank N makes ranks N+1 and below
moot for that construction.

| Rank | Gate | What it proves | How |
|---|---|---|---|
| 1 | **TRUTH** | The implementation matches its reference bit-exactly | Frozen vectors, cross-language harness |
| 2a | **Determinism** | Same input → same output, always | Frozen vectors recomputed |
| 2b | **Avalanche** | Single-trit input changes → every output digit changes at the neutral rate | Statistical sweep, ≥10k samples |
| 2c | **Diffusion** | Every output digit reachable by some input change | Union of changed positions over flips |
| 2d | **State-level** | The chaining value (not the output) diffuses properly | Same as 2b but on internal state |
| 2e | **Tamper** | Any constant change → vectors fail | Automated constant flipping |
| 3 | **Resistance** | Survives stated attack budgets | Differential, slide, multicollision, probe batteries |
| 4 | **Economy** | Throughput / state size / cost | Benchmark (informational only) |

**Critical rule: output-level gates are NOT sufficient.** A chained
output expander can mask state-level diffusion holes (measured: a
construction with 42/96 state trits below floor showed 0/486 output
positions outside band). Always gate at the chaining value.

---

## The experiment lifecycle

```
SEED → REGISTER → BUILD → CROSS-VALIDATE → HARVEST → TOOLIFY
```

### SEED
Pick work by two filters: (a) can the current arsenal verify it cheaply?
(b) surprise × groundability. If verification is expensive, building the
verification tool IS the work.

### REGISTER
Pre-register 3+ falsifiable predictions in the ledger. Each prediction
states: the expected outcome, the number that decides it, and what a
falsification means. If you cannot state what would prove you wrong,
you don't have a prediction — you have an opinion.

### BUILD
Build the construction AND the instrument. Controls go INTO the
instrument, not beside it. Every table, cache, and intermediate result
gets a self-check that fails loudly on drift.

### CROSS-VALIDATE
Hand the work to a second agent or a second implementation (the
"stranger" pattern). The stranger sees only the published spec — never
the workspace. **Continue regardless of outcome**: a failed validation
is a new axis, not a block.

### HARVEST
Results append to the ledger with budget labels. Failures are kept —
they are the map. Every non-grokking cell gets classified: dead run,
slow grok out of budget, or genuinely unlearnable.

### TOOLIFY
After each cycle, extract the reusable artifact: the pattern, harness,
gate, or loader that made verification cheap this time. **Never ship
a result without asking what tool it leaves behind.**

---

## The instrument bug taxonomy

Six bugs caught this session. Every one was found by a control or a
cross-check. Every one would have produced a WRONG scientific result
if not caught.

| # | Bug | Symptom | Prevention |
|---|---|---|---|
| 1 | Duplicate inputs counted as collisions | "Collisions" appear in the blake2b control | Dedupe inputs; carry a reference-hash control |
| 2 | Wrong denominator (divided by position-0 count) | Rates > 1.0 (impossible) | Track the trial count explicitly; assert rates ≤ 1 |
| 3 | Stale bytecode in tamper tests | Tampers "INVISIBLE" but actually ran an older module | Unique file path + unique module name + `dont_write_bytecode` |
| 4 | Duplicate (msg, tick) pairs in injectivity probes | False positive collisions | Dedupe the input PAIR, not just the message |
| 5 | pgrep self-match in process watchers | Process count never reaches zero | Bracket-pattern: `[t]rain` not `train` |
| 6 | Packed-integer arithmetic ≠ field operations | Distributivity fails, constants shift | Use algebraic operations (f36_add/mul), never integer ops on packed representations |

**Meta-rule:** when an instrument produces an absurd output (rates > 1,
zero always, identical results across different inputs), treat it as
an instrument failure FIRST and a finding SECOND. Investigate the
instrument before reporting the result.

---

## Structural findings (what kills custom hashes)

These are measured, not theoretical. Each was found by a specific
instrument, and each has a prevention.

### Invariant subspaces
**Cause:** a linear layer mixes groups of stride s; an affine permutation
with slope a maps each residue class mod (n/s) into a single class.
**Result:** the permutation decomposes into n/s invariant subspaces that
no round ever mixes. Avalanche collapses to ~1/(n/s).
**Prevention:** verify transitivity of the group generated by (linear
layer support, permutation) at build time, exhaustively.
**Measured:** T1-v1 had 9 subspaces; avalanche 0.074 (seq 34).

### Dead pipes
**Cause:** a "dual pipe" architecture where one pipe is initialized
but never written by message data. Output statistics look perfect
because the expander draws from the live pipe only.
**Prevention:** state-level diffusion gate on every state word; verify
that every state component receives message-dependent input.
**Measured:** T2-v1 pipe B never written at any round count (seq 43).

### Expander masking
**Cause:** a chained output expander (`c = f(c, w[j])`) saturates
change probability toward 1 regardless of the core's quality. The
output ruler cannot see state-level holes.
**Prevention:** always gate the chaining value, not just the digest.
**Measured:** T4 at 2 passes: output 0/486 outside band, state 42/96
below 0.6 (seq 44).

### Ring arithmetic kernels
**Cause:** in Z/(p^k), the p-adic filtration creates structured kernels:
f(x + p^j) = f(x) for a fixed fraction of inputs, determined by the
coefficient divisibility.
**Prevention:** use field arithmetic (GF(p^k)) or explicitly break the
divisibility chain in the round function.
**Measured:** GF(3^6) kernel 0/1000 (correct); Z/729 kernel 1/3 of
inputs at 2/3 of positions (seq 46, 74).

### Modular domain tags
**Cause:** a domain tag encoded in a fixed-width modular field wraps
at the modulus, creating replay-congruent inputs.
**Prevention:** encode domain tags in variable-width digit strings with
unambiguous terminators, or hash-bind them.
**Measured:** tick replay at distance 729 (v2), then 729² (v3), then
structurally closed (v4) (seq 37, 53, 100).

### Packed-representation confusion
**Cause:** treating packed-integer representations of algebraic elements
as if integer operations correspond to algebraic operations. Balanced
ternary digits packed into non-negative integers do NOT support integer
addition as field addition (carry structure differs).
**Prevention:** always use algebraic operations (f36_add, f36_mul) on
unpacked representations; never integer ops on packed values.
**Measured:** u64-as-i64 constant shift (seq 67); integer-add vs field-add
distributivity failure (seq 122).

---

## The representation principles

From the grokking-adjacent experiments:

1. **Tokenization dominates group structure.** Multi-token small-vocab
   encodings fail to memorize regardless of the underlying algebra
   (seq 118: binary, ternary, and Zeckendorf all fail; single-token
   integer groks).
2. **Single-token operands fix the binding problem.** Once fixed,
   the group-order question becomes testable (E6).
3. **Ternary-weight speedup is real but regime-specific.** It exists
   in the hard, decay-gated regime; vanishes at the fast-grok limit;
   reverses at the hardness edge (lab F2/F12/F14).
4. **Quantization determines WHERE solutions live.** Ternary networks
   are forced into concentrated spectral pathways; fp spreads across
   nonlinear views (lab F17/F18).
5. **The neutral point depends on the alphabet.** Binary: 0.5. Ternary:
   2/3. Golden-mean digits: ~0.47. Derive, never reuse.

---

## General patterns (meta-level, from 123 records)

These cut across every experiment, bug, and finding. They are the
reasons the specific results look the way they do.

### Verifiability is the frontier
Routes that can be cheaply verified advance; routes that cannot stall.
This is not a limitation — it is the selection mechanism. The frontier
is defined by what the current arsenal can verify. Building tools that
make new things verifiable IS pushing the frontier. Each generation of
tools moves a class of claims from "hard to verify" to "easy to verify."

### The instrument is always the first suspect
Instruments fail more often than constructions do. When a result is
surprising, suspect the instrument before the construction. When a
result matches your prediction, suspect that the instrument confirmed
your bias. The only escape is controls that can fail independently.

### Confounds hide behind similarity
When two things produce the same observable, the FIRST question is:
what different experiment would separate them? The answer usually
involves measuring at a different level (state vs output, input-access
vs distribution-only, single-token vs multi-token). The biggest
mechanism discoveries came from disentangling things that looked
identical under the first instrument used.

### The medium outlasts the means
The specific computational operations change across every experiment.
The structural medium (pre-registration, append-only verification,
independent cross-checks, the gate hierarchy) persists and compounds.
Three systems built independently converged on the same constitution.

### Nulls are evidence — if budget-labeled
A null result (no attack found) is genuine resistance evidence, but
only because it says "survived N samples of class X." Without the
budget, a null is vacuum. With it, nulls from different instruments
map the coverage of the attack surface.

### Falsifications are directions
A confirmed prediction says "your model was adequate." A falsified one
says "your model was wrong in a specific, localizable way." The
falsifications drove every major revision in this program.

### Structure converges
Systems built independently converge on the same constitutional
structure (append-only chains, pre-registration, independent
verification). The convergence is evidence that this is the minimal
viable structure for honest automated research.

---

## Known gaps (registered, not assumed)

These are things the methodology does NOT yet cover:

- **No formal proofs completed.** Lean proofs are queued but not done.
  The program measures but does not yet prove.
- **No external attack.** Everything is self-attacked. The publish/
  bundle exists but no outsider has tried. This is the largest gap.
- **No cross-instrument calibration.** The relationship between neural,
  differential, and state-level instruments is mapped qualitatively
  (which sees what) but not quantitatively (what does a neural score
  of X mean in differential terms).
- **No known-broken external reference.** All instruments have been
  tested only against custom constructions. A reduced-round cipher
  with published attacks would calibrate sensitivity against a
  known ground truth.
- **No side-channel coverage.** All gates are mathematical. Timing,
  power, and cache channels are untested.

---

## The workflow protocol

Roles: **sovereign** (route selection, axioms, fitness), **observer**
(pre-registration verification, controls, harvest), **builders**
(construct + register), **strangers** (replicate from spec only).

Rules:
- Registrations are BINDING across agent death and replacement
- Strangers see the spec, never the workspace
- Corrections are new records naming what they correct
- The ledger is the only memory
- Progression and the means of progressing advance together
- A tool that increases verification throughput is always worth building

Scaling: the swarm advantage grows with agent count on legible work
(measurement, verification, ports, sweeps) and stays flat on question
selection (sovereign's role). Pre-register the scaling hypothesis and
measure it.

---

## The claims language

| Claim | Meaning |
|---|---|
| "Confirmed" | Prediction matched measurement within stated parameters |
| "Falsified" | Prediction did not match — kept, with the revision it forces |
| "Survived budget X" | No attack in class X found within N samples/steps |
| "UNKNOWN beyond" | No claim made past the tested budget |
| "Instrument failure" | The measurement tool is wrong — result void |
| "Deviated" | Registration was wrong; the deviation itself is informative |

---

## Example experiments (all pre-registered, all harvested)

| ID | Question | Outcome |
|---|---|---|
| E1 | Can a neural net distinguish broken from fixed constructions? | Yes (1.0000 on broken); no (chance on fixed, all rounds) |
| E2 | Does representation dominate group structure? | Yes — trit tokens delete the phenomenon |
| E3 | Are reduced-round cores learnable? | No (null at all budgets — genuine resistance signal) |
| E5 | Does Zeckendorf tokenization inherit memorization-resistance? | Yes — and plain binary does too (alphabet, not constraint) |
| E6 | Is golden-linear structure learnable in ternary means? | Running |

Each took 1-3 hours of agent time and produced a mechanism-level
finding that narrowed the design space.
