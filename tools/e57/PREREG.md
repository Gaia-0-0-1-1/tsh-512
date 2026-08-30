# E57: categorical dictionary break/repair replay

Pre-implementation registration on 2026-08-30. This is a **confirmatory,
artifact-bound replay**, not a blind discovery experiment: an exploratory
read-only probe already observed the cyclic-permutation break and inverse
repair. E57's evidentiary value is fixed execution, exhaustive assertions, and
durable receipts.

## C

For the two fixed E36 component circuits, composite success at the continuous
logit-to-embedding join is causally sensitive to the categorical
class-coordinate/embedding-row dictionary. Breaking only that dictionary will
break exact composition; applying the known inverse dictionary will restore
the original logits and exact function without changing either component.

Shape: finite exhaustive replay over two 4,096-input domains. Instrument:
frozen components, one fixed derangement, nine temperatures, four join arms,
tensor identities, exact prediction fingerprints, and fail-closed assertions.

## pred

At T=1, in both directions:

- baseline equals the complete truth vector;
- the wrong dictionary is not exact;
- explicit inverse repair equals truth and its logits are bit-identical to
  baseline;
- gauge-equivalent embedding-row repair equals truth and its logits are close
  to baseline at rtol 1e-5 / atol 5e-5.

The prediction loses if either fixed component fails admission, either
baseline is non-exact, the wrong dictionary remains exact, explicit repair is
not bit-identical, gauge repair is non-exact at T=1, or any required artifact
or cell is missing. A component-admission failure is reported separately and
does not license retries.

## Fixed component instruments

- `Z4xZ2`: seed 0, exactly 5,400 CPU updates.
- `Z2x2x2`: seed 0, exactly 3,600 CPU updates.
- E36/E20 architecture: d64 `TinyTransformer`, `phi1` lattice.
- E36 optimizer/data: 80/20 task split, AdamW lr 1e-3, wd 0.5,
  betas (0.9, 0.98), batch 64.
- No accuracy is measured until the fixed terminal step. Each component must
  then be 64/64 exact. No retry, replacement seed, extra step, or selected
  checkpoint is permitted.

The tasks, seed, horizons, and expected exactness are historically selected
from E36 and the exploratory probe. They are instruments for the interface
assay, not new evidence about grokking frequency or training dynamics.

## Interface protocol

- Pairs: `Z4xZ2(Z2x2x2)` and `Z2x2x2(Z4xZ2)`.
- Temperatures: 64, 16, 4, 1, 0.5, 0.25, 0.1, 0.05, 0.01.
- Fixed derangement `pi(old -> new) = [1,2,3,4,5,6,7,0]`, with
  `inv[pi] = arange(8)`.
- For inner probabilities `p`, construct `q = p[:, inv]` so
  `q[:, pi[i]] = p[:, i]`.
- Baseline: `p @ outer.embed.weight`.
- Wrong dictionary: `q @ outer.embed.weight`.
- Explicit repair: `q[:, pi] @ outer.embed.weight`.
- Gauge-row repair: `q @ outer.embed.weight[inv]`.
- Enter the frozen outer transformer through E36's faithful `b_stack`.

For every temperature, explicit repair logits must be bit-identical to
baseline and gauge logits must be close. Baseline/repair/gauge must equal the
complete truth vector for all T <= 1. The wrong arm is measured everywhere;
only non-exactness at T=1 is required. Do not canonize a particular wrong-arm
accuracy.

## Verdicts

- `PASS_DICTIONARY_BREAK_REPAIR`
- `COMPONENT_ADMISSION_FAIL`
- `INSTRUMENT_FAIL`
- `BREAK_NOT_OBSERVED`
- `GAUGE_NUMERICAL_DRIFT`
- `INCOMPLETE`

## obl

- mechanical: permutation identities, domains, `b_stack` faithfulness,
  component admission, tensor equality, result-chain and manifest hashes.
- bounded: exactly two selected circuits, two directions, one already-seen
  derangement, and the registered temperature grid.
- conceptual: this can establish categorical-interface sensitivity for these
  instruments only. It cannot establish group-specific geometry, prevalence
  across seeds/permutations, systematic OOD composition, or a grokking
  mechanism. A genuinely prospective follow-up must use unseen permutations,
  automorphisms, output conjugations, tasks, and seeds.

