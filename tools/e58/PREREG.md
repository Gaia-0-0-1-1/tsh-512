# E58: repaired hyperbyte router, prospective CUDA assay

Pre-implementation registration on 2026-08-30. Seeds 5700-5703 were searched
for in the repository before selection and had no prior use. No E58 training,
accuracy probe, or router-gradient execution occurred before this document.
The E20 source defect itself is already known and recorded separately in
`E20_ROUTER_AUDIT.md`; the performance result of the repair is prospective.

## score / back

Timeline sequence 168's router-specific conclusion is superseded by sequence
265: E20's detach severed every router gradient. Historical E20 terminal rows
remain outcomes of an untrained stochastic-mixture arm, not learned routing.

## C

A forward-identical repair of E20's straight-through estimator makes its
per-matrix lattice router trainable. At matched initialization, data, update
budget, Gumbel schedule, and nominal parameter count, a live router can then be
distinguished from fixed representation, fixed heterogeneous representation,
the legacy dead router, and ordinary optimization/regularization effects.

Shape: finite bounded experiment. Instrument: two order-8 tasks, four unseen
seeds, six main arms, two shuffled-label controls, fixed horizons, complete
train/test evaluations every 200 updates, raw router telemetry, exact source
and schedule hashes, and fail-closed artifact reduction.

## pred

P1, mechanical graph gate: on disposable matched clones and the same explicit
Gumbel tensor, legacy-dead and repaired-live logits are bit-identical. All
seven dead router gradients are `None`; all seven live gradients are present
and finite with positive aggregate norm. One optimizer step changes live
router logits and leaves dead logits unchanged. Any miss is
`INSTRUMENT_BROKEN`; no performance result is admitted.

P2, prospective primary: on Z8, `LIVE_WD0` hard-routing evaluation beats
`LEGACY_DEAD` on the registered capability score without a lower memorization
count. It loses if it does not. A stronger adaptive-routing result additionally
requires beating `PHI1` and `HETERO_FIXED` and using at least two hard lattice
choices across the seven matrices in at least one successful seed.

P3, optimization floor: on Z2x2x2, every main arm should reach sustained exact
train fit in at least three of four seeds. Failure identifies an arm-specific
optimization floor and blocks a representation interpretation for that arm.

P4, leakage control: neither FP nor `LIVE_WD0` reaches the sustained exact test
gate on fixed point-shuffled Z2x2x2 labels. Any such gate yields
`INVALID_LEAKAGE_CONTROL`.

Router weight-decay 0.5 versus 0 is a registered mechanism contrast. If only
`LIVE_WD0` moves decisively away from uniform or wins, ordinary AdamW
regularization—not representation alone—explains the difference.

## Arms and matching

Tasks and fixed updates:

- Z2x2x2: 12,000.
- Z8: 20,000.

Seeds: 5700, 5701, 5702, 5703. No retry, replacement, extension, or
test-triggered stopping.

All arms use the E20 d64 one-layer transformer, train fraction 0.8, AdamW
lr 1e-3, core weight decay 0.5, betas (0.9, 0.98), and the complete 51-point
training set on every update. The same core state is cloned across arms for
each task/seed. Nonrouter arms carry 21 unused shadow scalars so every arm has
50,709 nominal parameters. Equal update budget is not equal FLOPs; runtime is
not a scientific endpoint.

1. `FP`: full-precision matrices.
2. `PHI1`: every matrix on the fixed five-state phi1 lattice.
3. `HETERO_FIXED`: q/k/v/o/w_in/w_out/unembed assigned
   ternary/phi1/phi2/ternary/phi1/phi2/ternary.
4. `LEGACY_DEAD`: E20's detached soft-Gumbel mixture, with explicit shared
   noise and an untrained router.
5. `LIVE_WD05`: forward-identical gradient repair; router weight decay 0.5.
6. `LIVE_WD0`: the same repair; router weight decay 0.

All three lattice tables are registered device buffers. Training uses tau-1
soft Gumbel mixtures. For each task/seed, a float32 Gumbel schedule of shape
`[steps, 7, 3]` is generated on CPU from the registered seed rule, hashed, and
shared exactly by all router arms. Evaluation samples nothing: router arms are
evaluated both with `softmax(router_logits)` and with hard argmax routing.
A soft-only win is convex-mixture evidence, not categorical base-id evidence.

The exact repair is:

```python
dead = weight + (w_eff - weight).detach()
live = dead + (w_eff - w_eff.detach())
```

The second term is numerically zero and restores only the router derivative.

## Metrics and grokking definition

Evaluate the complete 51-point train and 13-point test partitions at step 0
and every 200 updates. A gate is sustained only after five consecutive
qualifying evaluations:

- memorization: train accuracy >= 0.995, which means 51/51 exact;
- generalization: test accuracy >= 0.95, which means 13/13 exact.

Record gate onset and confirmation. Operational delayed grokking additionally
requires generalization onset at least 1,000 updates after memorization onset.
Immediate or concurrent exact train/test success is ordinary generalization,
not grokking.

The primary Z8 capability score is lexicographic: sustained-generalization
count across four seeds, then lower restricted-mean generalization confirmation
step, with nongeneralizing cells censored at 20,800. A score beats another if
its count is larger, or if counts tie and its restricted mean is at least 800
updates earlier. Memorization count is a guardrail. Delayed-grok counts and
delays are reported separately; they never reward a slower method over an
immediately generalizing one.

Telemetry at every evaluation includes train/test loss and accuracy, soft and
hard router evaluation, per-layer logits/probabilities/entropy/argmax,
router-gradient norm, cumulative router delta, route diversity, core norms,
and fingerprints for initialization, split, labels, and Gumbel schedule.

## Control and verdict reduction

The shuffled control uses one fixed permutation of the complete Z2x2x2 label
vector, preserving the histogram exactly, with the ordinary per-seed split.
Only FP and `LIVE_WD0` run this control.

Verdict precedence:

1. missing/tampered cell or artifact -> `INCOMPLETE`;
2. failed graph/determinism/forward-parity gate -> `INSTRUMENT_BROKEN`;
3. shuffled-label sustained test gate -> `INVALID_LEAKAGE_CONTROL`;
4. shared easy-floor failure -> `OPTIMIZATION_FLOOR`;
5. live hard win over dead+fixed comparators with diverse routes ->
   `ADAPTIVE_ROUTER_BENEFIT`;
6. live hard win with a single global route -> `GLOBAL_CHOICE_BENEFIT`;
7. live soft-only win -> `MIXTURE_ONLY_BENEFIT`;
8. fixed heterogeneous beats phi1 while live does not ->
   `HETEROGENEOUS_REPRESENTATION_ONLY`;
9. otherwise -> `NO_BOUNDED_ROUTER_BENEFIT`.

All raw outcomes remain reported even when an earlier verdict takes
precedence.

## obl

- mechanical: source/config hashes; exact arm/task/seed/control sets; graph
  liveness; forward equality; parameter counts; split/label/schedule hashes;
  chain, checkpoints, telemetry, manifest, and reducer replay.
- bounded: two tasks, four seeds, one architecture, one temperature, one
  random-point split family, and fixed horizons.
- conceptual: this cannot establish structural OOD generalization, universal
  router advantage, a group-theoretic mechanism, or frontier-scale grokking.
  The 1,000-step delay is an operational threshold on a tiny finite task.

