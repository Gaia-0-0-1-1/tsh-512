# E56 pilot: memorize, fork, then test

Registered locally before implementation and execution on 2026-08-29.
This file is the ledger for E56 because `timeline.jsonl` and `tools/e55/`
belong to an active E55 run and must remain untouched.

### Pre-execution addendum: legacy-protocol sentinel

A read-only implementation audit after registration found that E47 imports
E20's 50,688-parameter bias-free-MLP model, whereas E32 used a distinct
51,008-parameter local model. E56 therefore names and uses the **E47/E20 model
variant**. Before the fork assay it also runs the four hard cells from scratch
at E47's weight decay 1.0 for 20,000 fixed updates, logging train as well as
test behavior. This is an E47-configuration diagnostic, not an exact replay:
E47 ran 50,000 CPU updates with the post-initialization global RNG, whereas E56
runs 20,000 CUDA updates with a dedicated recorded batch RNG. This sentinel
was added before any E56 execution.

The same audit found that the legacy random point split leaks nearly all
commutative/associative symmetry orbits between train and test. E56 retains
that split solely to diagnose E47 on its own terms and records this limitation;
no result is promoted as structural OOD evidence. An orbit-disjoint assay is a
separate future obligation.

## C

The four E47 hard composite cells exhibit a post-memorization generalization
barrier rather than an inability of the d64 one-block model to fit its training
set.

Shape: bounded finite experiment. Instrument: full train/test accuracy curves,
fixed train-only admission gate, fixed horizons, and independent runs without
retries or test-triggered stopping.

## pred

At least three of the four hard task families will halt at
`no_memorization_within_budget` in both preregistered seed streams because the
E47 result is primarily an optimization/capacity ceiling at this
configuration. This prediction loses if at least three families sustain train
accuracy >= 0.995 in both seeds. Tasks are paired within a seed; the eight
task-seed cells are not treated as eight independent population replicates.

## Protocol

- Reuse E47's exact four hard tables, composite convention, d64
  `TinyTransformer`, 80/20 split, AdamW learning rate 1e-3, betas (0.9, 0.98),
  and batch size 64.
- Pilot seeds are exactly 0 and 1. There are no retries or replacement seeds.
  The same split/init/batch seed is deliberately paired across task families;
  independence is asserted only between the two seed streams.
- Phase A uses weight decay 0.0 for at most 20,000 updates.
- Before Phase A, the four hard tasks receive a fixed 20,000-update
  from-scratch E47-configuration sentinel at weight decay 1.0.
- The operational train gate requires full-train accuracy >= 0.995 at five
  consecutive evaluations 200 updates apart. This is sustained near-perfect
  fit (at most 16/3,276 errors), not literal zero-error memorization. Test
  measurements are logged but cannot stop Phase A.
- Once the train gate is confirmed, clone the exact model and optimizer state
  into weight-decay branches 0.0, 0.5, and 1.0. Give every branch the same
  minibatch stream and exactly 20,000 further updates.
- Grokking requires train accuracy >= 0.995 and held-out accuracy >= 0.95 at
  five consecutive evaluations. No branch stops early.
- Matched controls: the E47-easy structured composite T8(T8), and a fixed
  label-shuffled Z4xZ2(Z4xZ2) table on the same 4,096 inputs.
- CUDA is mandatory and PyTorch CPU threads are fixed to one while E55 runs.
- Save every evaluation, terminal classification, environment/config hashes,
  and memorization/final checkpoints. Never overwrite an existing run.

## Interpretation

- No train gate: `NO_MEMORIZATION_WITHIN_BUDGET`; this is not failed grokking.
- Train gate, then later sustained test gate: `POST_MEMORIZATION_GROK`.
- Train and test gate first-qualifying steps separated by at most 800 updates
  (the span between five evaluations at cadence 200):
  `ORDINARY_GENERALIZATION`.
- Train gate without test gate: `MEMORIZED_NO_GENERALIZATION`.
- If a branch confirms a grok gate but its final evaluation falls below either
  threshold, append `_THEN_FORGOT`; retain both the historical gate and the
  terminal loss of criterion.
- If Phase A already confirms both gates before the fork, classify that event
  as `PRE_FORK_ORDINARY_GENERALIZATION` and report whether each continuation
  retains or forgets it; do not relabel it as nongeneralization merely because
  a post-fork window is short.
- Shuffled labels reaching a test gate invalidates the split or measurement.
- If T8(T8) does not memorize in both seeds, the aggregate hard-cell verdict
  remains open because the structured training-floor control failed.
- The aggregate prediction is scored by task family across its paired seeds,
  never by treating the eight cells as independent Bernoulli trials.
- A missing preregistered task-seed cell makes the aggregate verdict
  `INCOMPLETE`; absent cells are never counted as failures.
- This two-seed pilot can choose the next assay; it cannot establish a
  population-level universal.

## obl

- mechanical: implementation self-tests, artifact hashes, checkpoint/curve
  receipts.
- bounded: two seeds across all four hard cells and two controls.
- conceptual: only after the pilot, choose between capacity sweep,
  post-memorization replication, or interface causality. No conclusion is
  promoted beyond the executed cells.
