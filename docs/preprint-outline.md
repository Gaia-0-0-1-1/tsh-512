# Grokking Composes: The Wall, the Free Monoid, and the Semantic Join

**Working title (candidate list):**
1. *Grokking Composes* — the arc's one-word thesis with its refutation
2. *The Fusion Wall is an Optimization Wall* — the H4 law as headline
3. *Learn the Phonemes, Compose for Free* — the operational takeaway

**Status:** outline, pre-registration-complete; every claim below is
already measured and recorded on the TSH-512 timeline (seq numbers in
brackets). Nothing in this paper is argued — every sentence of the
results section is a ledger record.

## Abstract (the arc in five sentences)

Modular addition and its algebraic kin grok: small transformers
trained on Cayley tables transition from memorization to exact
generalization [E7/E20]. But composition is a wall: composites of
grokkable functions are themselves mostly NOT learnable from scratch
(0/32 hard-label, 0/8 even with a correct teacher's soft targets)
[E32, E36-B]. Yet composition is FREE at the function level: the
circuit vocabulary is closed under composition — a free monoid, 16
distinct composites derivable from 4 banked phonemes with zero
training [E25]. We map four routes across the wall — sequential
wiring (exact, 2.8x cost) [E33], the fold (truth tables as circuits,
131,090x) [E34], the tree (no depth limit) [E35], and the semantic
join: a ZERO-PARAMETER weight fusion, `softmax(inner logits) @
outer.embed`, exact on the full composite domain when the inner is
sharp [E36]. The result is a deployed learning loop that acquires
composite functions exactly while paying only for missing phonemes —
compounding across every future target that reuses them [E37].

## 1. Introduction — the question

- Grokking literature: algorithmic tasks, modular arithmetic,
  representation learning (cite Power et al. 2022; Nanda et al.
  2023 modular addition; Gromov 2023; Kumar? — the lab's lit review
  has the canonical set).
- Our angle (not in the literature — verified, grokking-hash-design
  memory): composition. What happens when the TARGET is a composite
  of two grokkable functions?
- The measured answer splits into a negative and a positive result,
  and the positive is an architecture.

## 2. Setup — the substrate (all previously measured)

- Tasks: 7 verified group structures (Z8, Z4xZ2, Z2x2x2, Q8, D4,
  S3, Z7), exhaustive Cayley verification [E6].
- The grok config: d64 TinyTransformer, train_frac 0.8, wd 0.5,
  AdamW, 20k cap, phi1 lattice (lattice detail cite E20 — per-matrix
  routing falsified, spectrum global).
- The learnability ladder as measured: Z2x2x2 easy (~4k steps),
  Z4xZ2 grokkable (~5k), Z7/Z8 marginal, Q8/D4/S3 walls (0/20) [E20,
  vocabulary WALLS table].
- Fingerprinting: extensional identity = hash of outputs on the full
  domain; function-deterministic [E22].

## 3. The negative result — the fusion wall (E32, E36-B)

- P1: 16 pairwise composites trained from scratch at the parts'
  config: 7/32 grok (vs parts grokking normally). Composites with
  Z8/Z4xZ2 outer: 0/16. Z2x2x2 outer: 0/8 [E32].
- The placement asymmetry (P2): inner cleanliness matters [E32].
- The deeper negative (NEW, E36-B): distillation from the CORRECT
  wired teacher — soft targets, T=2/T=10, KL+CE — 0/8 grok, best
  0.88. The wall is function-level, not signal-level [E36].
- Control arm replication: E32's originals (0.22–0.88 partial
  plateau) vs H4's hard arm (0.40–0.46) — same regime [E36].

## 4. The positive result — four routes across the wall

1. **Sequential wiring (H1):** A's argmax -> mod -> B's input.
   16/16 exact [E33]. Cost: 2.8x.
2. **The fold (H2):** composite truth tables as first-class circuits.
   Identity preserved (16/16 fingerprint match), 131,090x speedup,
   zero training [E34].
3. **The tree (H3):** fold-of-folds. No depth limit — the honest
   correction (the pre-registered n^8 crossover prediction was
   WRONG: input space stays n^4 under pair-of-pairs) [E35].
4. **The semantic join (H4):** `softmax(inner(x)) @ outer.embed`
   entered at the outer's embedding — zero parameters, zero
   training, EXACT on the full 4096-point domain, both directions
   [E36]. The temperature sweep measures the softmax
   non-homomorphism: exact for T<=1, 0.98 at T=4, collapse at T=16.

## 5. Why the wall exists — the geometric law (E36-C)

- Sharpness: grokked circuits emit near-one-hot logits (margins
  min ~1.5, mean ~16–18) [E36].
- Tolerance: grokked circuits READ near-exact embedding rows —
  accuracy collapses at embedding noise sigma ~10–20% of the mean
  embedding norm (Z4xZ2: collapse at 0.05/0.397; Z2x2x2: survives
  to 0.1–0.2/1.054) [E36].
- The law: the semantic join lands ON the manifold (free); learned
  joins must land IN the tolerance ball, and the composite's loss
  landscape never guides them there. Localization experiment (A3):
  both parts frozen, only an 8x64 join trained — groks in 400 steps
  when the outer is wide-tolerance (Z2x2x2), dies at chance when
  the outer is tight (Z4xZ2) [E36].
- The softmax non-homomorphism is real but SECOND-ORDER (exact at
  grokked sharpness; bites only when softened).

## 6. The deployed loop (E37)

- The loop: query (MISS) -> grok ONLY missing phonemes -> FUSE
  (free) -> FOLD (bank, dedup by fingerprint) [E37].
- P1: exact composites at phoneme-only cost (9000 steps total) [E37].
- P2's honest twist: cross-episode compounding — the second target
  cost ZERO steps (both phonemes already banked); the prediction's
  letter failed because the economy is better than predicted [E37].
- P3: direct-training controls fail (0/2, test 0.40/0.44) where the
  loop is exact [E37].
- The economy table: HIT free / COMPOSED free / FUSE
  free-after-phonemes / grok paid-once-per-phoneme.

## 7. Related work (to write)

- Grokking canon (Power, Nanda, Gromov).
- Circuit compositionality / function algebras (the tryte-vm's
  foldTables; truth-table composition is classical).
- Distillation (Hinton et al.) — our negative result is a boundary
  condition: distillation cannot cross the composition wall.
- Grokking literature on modular arithmetic specifically.

## 8. Limitations — the honest frontier

- Scale: d64 transformers, order-6/7/8 groups, CPU-measured. The
  wall's shape at depth/scale is unmeasured (E16 phi2-on-12B queued).
- The wall set is small: 4 grokkable phonemes, 16 composites.
- The semantic join is verified at pair composition; the tree (H3)
  extends to depth-3 tables but the JOIN-of-joins (semantic join
  applied recursively to already-fused models) is NOT yet measured —
  a live open question.
- Unattacked = UNKNOWN: no security claims anywhere (the TSH-512
  law); this paper is about learnability, not cryptography.

## 9. The preprint's pre-registered spine

Every results-section sentence maps to a timeline seq: E32 [197],
H1 [200], H4 design [206], H4 verdict [207], the loop [210]. The
paper is an EXTRACT of the ledger, not a narrative constructed
after the fact. (This is itself a methodological claim worth one
paragraph: pre-registration at the experiment level, hash-chained,
append-only.)

## Appendix candidates

- A: the 27-record E36 results table + the 20-record E37 table.
- B: fusion_viable() — the sharpness x tolerance criterion as code.
- C: the full temperature and sigma sweep curves.
- D: the free monoid census (16 composites from 4 phonemes, E25).

## What is NOT in this paper (scoped out)

- The hash-design program (TSH-512's origin) — separate track.
- The phi/metallic-means substrate — separate track (though E20's
  phi1 lattice is the config used here, so one bridging paragraph).
- The homeostat/economic DAG — the loop's controller, future work.
