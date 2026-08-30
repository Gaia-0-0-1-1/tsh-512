# E58 static audit: E20 did not train its router

Audit date: 2026-08-30. Base commit:
`7345a8b391d81de9ed5a8a9179979835bff2c04e`.

This is a source-level mechanism finding, not a performance result.

## Claim scored

Timeline sequence 168 concluded that E20's per-matrix lattice router added no
capability. That conclusion does not follow from the executed instrument,
because the router was disconnected from the loss.

## Mechanism

In `tools/e20/hyperbyte_test.py`, `effective_weight()` constructs `w_eff`
from `router_logits` through Gumbel-softmax. The forward pass then uses:

```python
w = self.weight + (w_eff - self.weight).detach()
```

The detach removes the complete derivative path from the loss through
`w_eff` to `router_logits`. The latent weight retains the identity
straight-through gradient, but each router parameter has `grad is None`.
Because router logits start at exactly zero and AdamW skips parameters with
no gradient, the logits cannot move.

At the same time, `F.gumbel_softmax(..., hard=False)` samples on every
forward, including evaluation. E20's `hyper` arm was therefore an untrained,
per-forward stochastic convex mixture, not learned categorical lattice
selection. Its sampling also shared the global Torch RNG with minibatch
permutations. The cached lattice tables are plain CPU tensors, making that
historical implementation unsafe to move to CUDA without repair.

## Resolution mark

Supersede the router-specific interpretation in timeline sequence 168:

- the 70 recorded terminal outcomes remain historical outcomes of the code
  that ran;
- they do not measure learned per-matrix routing;
- they do not establish that lattice choice is globally uniform or that a
  working router adds no capability;
- the fixed-lattice arm outcomes remain usable within E20's other stated
  split, seed, and early-stopping limits.

The corrected E58 assay is prospective for routing performance. Its graph
gate will mechanically confirm the dead/live gradient distinction before any
main result is admitted.

## Receipts

- `tools/e20/hyperbyte_test.py` SHA-256:
  `95471ebb3db19108d5d75525209c3c93d5da044228946b8ae0c81dfec041a6f2`
- `tools/e20/results.jsonl` SHA-256:
  `4a8abceaa130fa8b890dfbb8a2ae9b2903a6bccbf7a2787e9ddc02070fb8c97e`
- `tools/e6/math_structures.py` SHA-256:
  `1b4811259b1d177933ac0f667bba29919d8680e57c2ea437259e964afd5b0753`
- E20 preregistration: timeline sequence 160.
- Superseded router interpretation: timeline sequence 168.
