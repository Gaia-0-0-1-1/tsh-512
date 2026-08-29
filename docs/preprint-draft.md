# Grokking Composes: The Wall, the Free Monoid, and the Semantic Join

**Draft v0.2 — the results sections written from ledger records
(tsh-512 seq 197–220). Every quantitative claim cites its seq.**

## Abstract

Small transformers grok algebraic structure: trained on a group's
Cayley table, they transition from memorization to exact
generalization. We ask what happens when the target is a
COMPOSITION of grokkable functions, and measure a sharp asymmetry:
composites of GROUPS are mostly NOT learnable from scratch — 0/32
hard-label runs, and 0/8 even when distilling from a correct
teacher's soft targets [seq 197, 207] — yet composition is FREE at
the function level: the circuit vocabulary is closed under
composition, a free monoid where 16 distinct composites derive from
4 banked phonemes with zero training [seq 178]. We map four routes
across this fusion wall — sequential wiring (exact, 2.8x cost) [seq
200], truth-table folds (exact, 131,090x) [seq 202], the
depth-unlimited table tree [seq 204], and the semantic join: a
ZERO-PARAMETER weight fusion, `softmax(inner logits) @ outer.embed`,
exact on the full composite domain when the inner is sharp [seq
207]. The join is RECURSIVE — join-of-joins is exact at depth 2 in
all six tested orderings [seq 214] — and it composes into a
deployed learning loop that acquires composite functions exactly
while paying only for missing phonemes, compounding across every
future target [seq 210]. A control experiment extends the wall to
quantum gate groups: the Pauli group mod phase groks at its
isomorphism class's rate (200 steps, the fastest measured), but the
full Pauli group WITH phases walls at the same config where Z16
groks — the phase is a commutation carry, the same
order-dependent interaction term that blocks composite learning
[seq 217]. Finally, we measure the wall's BOUNDARY: composites of
monoids grok DIRECTLY in 200 steps where group composites wall —
the wall is the price of REVERSIBLE (information-preserving)
composition, not of composition itself [seq 220].

## 1. Introduction

The grokking literature (Power et al. 2022; Nanda et al. 2023;
Gromov 2023) has established delayed generalization on algorithmic
tasks, with modular arithmetic as the canonical case. Composition —
the target function being a composite of two individually grokkable
functions — is, to our knowledge, unmeasured. This paper measures
it and reports a two-sided result:

1. **A negative result with a deeper negative inside it.**
   Composites do not grok from scratch, and they do not grok even
   when a correct teacher provides soft targets. The wall is a
   property of the function, not the learning signal.

2. **A constructive result.** The wall is an optimization wall, not
   a representation wall: the composite is exactly representable
   and free to construct, through an operation we call the semantic
   join, which is itself recursively composable.

The practical consequence is an architecture: learn flat phonemes,
compose by construction, never train a composite.

## 2. Setup

All experiments use a fixed d64 TinyTransformer (4-token input,
single attention block, phi1 lattice quantization), AdamW lr 1e-3,
weight decay 0.5, train fraction 0.8, 20k step cap, CPU. Tasks are
Cayley-table classification: input (a, b), output a∘b. Groups:
Z8, Z4xZ2, Z2x2x2, Q8, D4, S3, Z7 (exhaustively verified), plus
Z16 and the Pauli tables of §6 (qiskit 2.4.2-verified). Identity is
extensional: the fingerprint is SHA-256 of the output vector on the
full input domain [seq 178]. Grok is defined as test accuracy ≥
0.95.

The learnability ladder at this config [seq 197, WALLS table]:
Z2x2x2 easy (~4k steps), Z4xZ2 grokkable (~5k), Z7/Z8 marginal,
Q8/D4/S3 walls (0/20 each).

## 3. The fusion wall (two experiments)

**E32 — hard labels [seq 197].** All 16 pairwise composites of the
4 banked tasks, trained from scratch at the parts' config: 7/32
runs grok (21.9%) vs parts grokking normally. Every composite with
Z8 or Z4xZ2 as the outer function fails (0/16 family); Z2x2x2 as
outer never groks (0/8). The placement asymmetry is measured: a
clean inner helps (Z2x2x2 inner: 2/8), a demanding outer kills
(Z8/Z4xZ2 outer: 0/16). Replication controls in the H4 runs land
in the same regime (0/4 grok, test 0.40–0.46; E32 originals
0.22–0.88 — a partial-generalization plateau below threshold).

**E36-B — soft targets [seq 207].** The decisive control: students
distilled from the CORRECT wired teacher (the H1 chain, verified
16/16) with KL loss on soft logits at T=2 and T=10, plus the true
hard labels. Result: 0/8 grok (best single run 0.88). A correct
teacher's soft targets do not cross the wall. The obstruction is at
the function level, not the signal level.

## 3b. The wall's boundary: group composites wall, monoid
composites are free [seq 220]

The scope of §3 is not "composition" but REVERSIBLE composition.
E40 measured non-group phonemes at the identical config:

- **The ladder:** bitwise monoids AND8/OR8 grok fast — 200 steps,
  exact both seeds — while QUASI8, an affine LATIN-square
  quasigroup (T(a,b) = (3a+5b) mod 8, bijective in both arguments,
  no associativity), WALLS (0/2, test 0.54/0.15). Latin-ness does
  not buy grokkability; the monoids' monotone idempotent structure
  does. The null control RANDOM8 never groks (chance) — grokking
  is structure extraction.
- **The wall arm:** AND8(AND8) and OR8(OR8) composites grok
  DIRECTLY FROM SCRATCH in 200 steps (test ≈1.0, both seeds) — the
  composite of a monoid with itself is easier to learn than QUASI8
  alone. Mixed composites carry: AND8(QUASI8) groks (1600–2000),
  QUASI8(AND8) groks (2400–4400) — the monoid component's
  information collapse frees the composite past the quasigroup's
  own wall. Only QUASI8(QUASI8) fails (0/2).
- **The join arm:** AND8(AND8) join is exact; OR8(OR8) join
  MISMATCHES at 0.9968 — one point off in 4096 — because OR's
  absorbing element (x|7=7) creates soft-margin inputs where the
  softmax mixture lands a hair off the embedding row. The join's
  sharpness precondition is violated at absorbing structures: the
  join is not representation-agnostic.

**The corrected law:** the fusion wall is the price of
information-preserving (reversible, Latin) composition. Group
outputs are uniform bijections — every token carries the full
computation's trace, and nothing collapses, so the flat learner has
no shortcut. Monoid outputs collapse (AND(x,0)=0 absorbs variance)
and the collapse IS the shortcut.

## 3c. The quantitative predictor: alignment, then entropy [seq 223, 224, 227]

E41 measured the hypothesis that the predictor is the inner's
output entropy. The retrospective over all 21 families CONFIRMS the
separation (mean inner entropy 2.705 bits for ever-grokked vs
2.952 for never-grokked; label entropy separates harder: 2.398 vs
2.9995) — and resolves E32's Z7 anomaly: the n=7 vocabulary caps
entropy at 2.807, below the 8-vocab wall's 3.0, which is why Z7
composites grok at all. But the DECISIVE prospective cell
falsified the entropy-causes-wall story: MOD6 — a high-entropy
(H=2.58 over 6 symbols), non-Latin, no-absorbing-element table —
has MOD6(MOD6) grok in 1200–1800 steps both seeds. High entropy
without bijection is learnable.

E43 then dissolved the intermediate band entirely. At 4 seeds per
family, the tunable cyclic tables T_k(a,b) = (a+b) mod k grok
their self-composites at EVERY entropy up to 3.0: T5 4/4, T6 4/4,
T7 4/4, and T8 (= Z8, full 3.0 bits) 3/4 — pooled with E32's
Z8(Z8) cells, 4/6. Yesterday's "coin-flip band" [2.58, 2.81] was
2-seed sampling noise. Self-composition of cyclic tables is
learnable at every measured entropy.

**The two-factor law [seq 227]:** the wall is governed by OUTPUT
ALIGNMENT first and entropy second. The decisive cell is PD — Z8
with outputs scrambled by a fixed permutation: same function, same
3.0-bit entropy, but 0/4 composites grok (test 0.22–0.63) where
T8 groks 3/4. The learner exploits the aligned successor structure
of cyclic outputs (the composite of a cyclic table is cyclic;
Fourier/periodic features line up), and scrambling the labels
destroys exactly that alignment. Entropy remains necessary — every
grokking composite in the corpus sits below 2.81 bits except the
aligned T8 — but the sufficient condition for the wall is full
entropy PLUS broken alignment. This is the E5 representation
taxonomy resurfacing at the composite level: the fusion wall is,
in its sharpest form, a REPRESENTATION wall. A third factor —
structural mismatch between outer and inner (E32's cross-structural
0/16 families) — is not touched by this sweep and remains open
(the E44 candidate). One more inversion the sweep surfaced: the T7
PHONEME fails to grok exactly (0/2, 12 attempts) while its
composite groks 4/4 in 1400–1600 steps — a composite can be easier
than its own parts (more averaging structure over 4 tokens than 2),
inverting the E32 framing at the phoneme level.

## 3d. The third factor: structural mismatch [seq 230]

E44 measured the cross grid T_k(T_j) at 4 seeds per family. The
grid is nearly empty: of 12 cross cells, only the ADJACENT-modulus
pairs grok at all — T5(T6) 4/4, T6(T7) 2/4 — and every
non-adjacent pair walls (0/4 or 1/4), even though both families in
every cell are CYCLIC (structurally identical up to modulus). The
inner's output range vs the outer's expected classes is the
mismatch variable: adjacent moduli nearly match, distant moduli
misalign badly. The E32 replication cells hold at 4 seeds
(Z4xZ2(Z2x2x2) 0/4, Z2x2x2(Z4xZ2) 0/4).

The scrambled-cross control inverted the E43 prediction: PD(T8) —
scrambled OUTER, aligned inner — groks 3/4 (2000–2400 steps), while
T8(PD) — aligned outer, scrambled inner — walls 0/4. The INNER's
alignment is what matters: the inner's outputs form the composite's
input-side interface, and scrambling them destroys the
representation the network needs, while a scrambled outer merely
renames the target classes. This echoes the E5 input-access
finding at the composite level.

**The three-factor law (final form):** composite learnability =
output alignment x inner-interface coherence x structural match,
with entropy as the water level beneath all three. Direct
composite learning fails on three independent measured factors;
the semantic join bypasses all three — it never trains the
composite.

## 4. Four routes across the wall

**H1 — sequential wiring [seq 200].** Chain the phonemes: A's
argmax → mod B's vocab → B's input. All 16 wirings compute exactly
the derived composite (16/16 fingerprint match). Cost: 2.8x a
single phoneme.

**H2 — the fold [seq 202].** Store the composite's full truth table
(all n⁴ entries) as a first-class circuit node, executable by
lookup. Identity preserved through the fold (16/16), 131,090x
faster than the wired chain. Zero training.

**H3 — the tree [seq 204].** Fold-of-folds: composites of
composites. Measured correction of our own pre-registered
prediction: the input space does NOT grow as n⁸ at depth 3 — the
pair-of-pairs convention keeps it at n⁴. There is no depth limit.

**H4 — the semantic join [seq 207].** The zero-parameter fused
model: `softmax(inner(x)/T) @ outer.embed`, entered at the outer's
embedding layer (its transformer stack run from the mixture). At
T=1 — the grokked inner's natural sharpness — this is EXACT on the
full 4096-point composite domain, in both directions of the
(Z4xZ2, Z2x2x2) pair. The temperature sweep measures the softmax
non-homomorphism directly: exact for T ≤ 1, 0.98 at T=4, 0.25 at
T=16, chance (0.125) at T=64.

**E38 — the join is recursive [seq 214].** The join-of-joins: the
fused pair (A→B) serves as the inner of a second join into C. The
fused pair's output logits are SHARP (margins min 2.5, mean 17.4 —
sharper than a raw phoneme's min ~1.5), which is why the recursion
works: all six orderings of Z4xZ2/Z2x2x2/Z7 are exact at 1.0000 on
the full 2401-point triple domain. The vocab interface generalizes
E25's mod convention to a continuous form (`vocab_map`, the matrix
M[i, i % n_to] = 1), with a discrete argmax-mod fallback where the
fold is many-to-one.

## 5. Why the wall exists: the geometric law [seq 207]

Three measurements compose into the law:

1. **Sharpness (the inner's property).** Grokked circuits emit
   near-one-hot logits: top1−top2 margins min ≈ 1.5, mean ≈ 16–18
   over the full pair domain.

2. **Tolerance (the outer's property).** Grokked circuits read
   near-exact embedding rows: accuracy collapses under Gaussian
   embedding perturbation at σ ≈ 10–20% of the mean embedding norm
   (Z4xZ2: collapse at σ=0.05, norm 0.397; Z2x2x2: survives to
   σ=0.1–0.2, norm 1.054).

3. **The join.** `softmax(inner) @ outer.embed` lands exactly ON
   the outer's embedding rows — inside any tolerance ball, zero
   parameters. Learned joins must land IN the ball, and the
   composite's loss landscape never guides them there.

The localization experiment (A3): both parts frozen, only an 8×64
linear join trained. It groks in 400 steps to 0.955 full-domain
when the outer is wide-tolerance (Z2x2x2) and dies at chance
(0.139) when the outer is tight (Z4xZ2). The obstruction is
geometric; the softmax non-homomorphism is real but second-order
(it bites only when the intermediate is softened, T ≥ 4).

**The H4 law:** the fusion wall is an optimization wall, not a
representation wall. Composites are exactly representable,
unlearnable from scratch (even taught), and free by construction.
The measurable criterion is sharpness(inner) × tolerance(outer).

## 6. The wall's sibling: the phase carry [seq 217]

The Pauli ladder extends the wall into quantum gate groups:

- **PAULI4** ({I,X,Y,Z} mod phase): groks in **200 steps**, both
  seeds — the fastest grok on this timeline. (Our pre-registration
  guessed cyclic Z4; construction revealed the Klein four-group —
  isomorphic to Z2x2x2, the ladder's easiest task, so the speed is
  consistent with its class, not new physics. The correction is
  kept.)
- **PAULI16** (the full 1-qubit Pauli group, ±1/±i × I/X/Y/Z,
  qiskit-verified): **walls** — 0/2 grok at 20k (test 0.765/0.490,
  the partial plateau).
- **Z16 control**: groks both seeds (4800/4200 steps). Order-16 is
  not the wall; the Pauli structure is.
- **PAULI16-FACTORED** (phase and Pauli labels decomposed into two
  heads): also walls (0/2, test 0.588 ≈ the pauli-only floor
  9/16 = 0.5625). The phase does not decompose into a free
  representation.

The mechanism: XY = iZ but YX = −iZ — same Pauli, different phase,
order of operands decides the carry. This is precisely the place-value
killer of the multiplication wall [E11/E12] and, we argue, the same
phenomenon as the fusion wall itself: composition products whose
outputs carry interaction terms (a phase, a cross-term) that the
flat learner cannot factor. The hypernetwork route — learn phonemes
flat, compose by construction — is the measured escape at both
levels.

## 7. The deployed loop [seq 210]

The loop: query → MISS (structured) → grok ONLY the missing
phonemes → FUSE (semantic join, free) → verify fingerprint → FOLD
(table circuit, banked with dedup by fingerprint).

- **P1 confirmed:** both target composites acquired exactly
  (fingerprint match, full 4096-point domain) at phoneme-only cost:
  5400 + 3600 = 9000 steps total. The composite is never trained.
- **P2's honest twist:** the pre-registered form (warm cheaper than
  cold per target) failed in the letter because the economy is
  better than predicted: the second target shared both phonemes
  with the first and cost ZERO steps — compounding across targets,
  not just re-queries. Total system cost for 4 episodes: 9000
  steps.
- **P3 confirmed:** direct-training controls on the same composites
  fail (0/2, test 0.40/0.44) at the identical config.

The vocabulary economy: HIT free / COMPOSED free / FUSE
free-after-phonemes / grok paid-once-per-phoneme, compounding
across every future target that reuses it.

## 8. Related work

To write: Power et al. (grokking), Nanda et al. (modular addition
circuits), Gromov (grokking beyond), Hinton et al. (distillation —
our §3 is a boundary condition: distillation cannot cross the
composition wall), truth-table composition (classical), quantum
circuit compilation (the genQC line — diffusion, not grokking).

## 9. Limitations

Scale: d64, order-4/7/8/16 structures, CPU-measured; the wall's
shape at depth and scale is unmeasured (E16 queued on GPU). The
phoneme set is small (4 grokkable, 16 composites). The semantic
join is verified at depth 2 [seq 214]; arbitrary depth is implied
by the sharpness-carry measurement but not measured beyond depth 2.
Unattacked = UNKNOWN: no security claims; this is a learnability
result.

## 10. Methodological note

Every results sentence above maps to a hash-chained, append-only
ledger record (tsh-512 timeline.jsonl, seq 197–220, chain verified
at each step). Predictions were registered BEFORE each experiment
(seq 196, 199, 202, 204, 206, 209, 213, 216); two pre-registered
predictions were WRONG (the H3 crossover, the PAULI4 cyclic guess)
and are reported as corrections in place. The paper is an extract
of the ledger, not a post-hoc narrative.

---
*Draft v0.2 assembled overnight (seq 212 outline → v0.1 → the E40 scope correction).
Next: related work, figures (the T sweep, the sigma sweep, the
ladder table), the E16 scale paragraph when the GPU frees.*
