# PROTOCOL — the recursive workflow (v1, extracted from what worked)

*The sovereign's doctrine (timeline seq 113) made operational. Every element
below is drawn from a measured success or kept failure in this session's
record — citations are timeline seq numbers. This file is itself dogfood:
it exists so the next experiment inherits the spine without rebuilding it,
and it will be revised by the same loop it describes.*

## The cycle

    SEED -> REGISTER -> BUILD -> CROSS-VALIDATE -> HARVEST -> TOOLIFY -> next

1. **SEED** — pick work by two filters, in order:
   (a) *verifiability*: can the current arsenal verify the result cheaply?
       If not, the route is wrong — either pick different work, or make the
       verification cheap first (that tool-build IS the work).
   (b) *surprise × groundability* (WHEEL_AND_FIELD_MAP's ranking).

2. **REGISTER** — pre-register falsifiable predictions BEFORE any
   measurement (seq 9-11, 26-29, 71-73, 99, E1-E5 phases). Registered
   after results = worthless. If an agent claims registration, verify the
   file exists before trusting it (seq 96's lesson). Registrations are
   BINDING on successors — a resumed or replacement agent inherits them
   and may not re-register (seq 102).

3. **BUILD** — std-only where the artifact is a core construction;
   controls built INTO the instrument, not beside it (the blake2b control,
   the broken-specimen control, the random-output control: seq 12, 98, E3).
   A control that has only ever passed has demonstrated nothing — plant
   defects to prove gates have power (tamper tests, variance_test.ps1).

4. **CROSS-VALIDATE** — hand the work to a second agent (the *stranger*,
   LadderLang's ceremony) or a second implementation (Python ↔ Rust,
   40/40). **Continue regardless of outcome**: a failed validation is a
   new axis, not a block — fix and re-run (F9's falsified flip-jitter
   hypothesis became far_rate; the u64-as-i64 modulus bug became a
   port-discipline rule). Agents validate faster than humans only when
   the tools make verification cheap — every tool should increase
   verification throughput.

5. **HARVEST** — results append to the ledger with budget labels; the
   strongest permitted claim is "survived budget X; unknown beyond"
   (seq 46, 82). Crashes are harvested too: agents die, the chain does
   not (seq 96, 102). Failures are kept — they are the map (law 7).

6. **TOOLIFY** — after each cycle, extract the reusable artifact:
   the pattern, harness, or gate that made this cycle's verification
   cheap. Recorded lineages: timeline.py → wave/marker orchestration →
   crash recovery (seq 92, 96, 102); the pycache-safe variant loader
   (seq 32) reused in every later instrument; port tests → family truth
   → bundle self-verification (seq 16, 67, 88). Progression and the
   means of progressing advance together — never ship a result without
   asking what tool it leaves behind.

## The roles

| Role | Holder | Load-bearing duty |
|---|---|---|
| **sovereign** | Owen | route selection, taste, the axioms (H, the fitness function) — the only role that does not scale away; it scales up |
| **observer/orchestrator** | steering agent | verifies registrations precede results, controls ran first, harvests crashes, holds the marker protocols |
| **builders** | swarm agents | SEED→BUILD under binding registrations |
| **strangers** | isolated agents | see only the published spec; replicate a price, never extend the substrate (Mint/Desk boundary) |

## The consistency rules (what makes subagent validation work)

- Work → subagent replicates → continue either way. Failure = new axis.
- Strangers see the spec, never the workspace (transcript-checked, as in
  the tick-ceremony R5 stranger arm).
- Corrections are new records naming what they correct (seq 83, 105).
- Registrations are binding across agent death and replacement.
- One serialization per claim; the ledger is the only memory.

## The edge-of-chaos control

Pre-registration holds the ground; the swarm explores. The ledger makes
every excursion recoverable, so exploration is cheap and wrong turns are
*profitable* (each kept failure maps an axis). The loop's speed on
legible work compounds with agent count; question-selection stays with
the sovereign (seq 112's measured boundary).

## Open protocol questions (registered, not assumed)

- P-W1: does the stranger-verification pass rate improve as the arsenal
  grows? (Measure: replication failures per cycle over time.)
- P-W2: what fraction of cycles TOOLIFY successfully? Below ~half, the
  recursion is not compounding — diagnose before scaling agents.
- P-W3: optimal wave size vs marker contention (we have run 1-3 agents;
  the curve at 10+ is unmeasured).
