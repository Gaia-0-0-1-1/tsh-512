# RESEARCH DAG — the skill tree across both programs

*Two research programs, one dependency graph: the TSH-512 hash timeline
(this repo) and the grokking lab (`spark:~/workstation/lab/`), plus the
bridge experiments (E1–E4) that connect them. Every arrow is "depends
on". Dashed arrows are the cross-cutting discipline (pre-registration,
controls) that feeds every measurement. Rendered with Mermaid.*

```mermaid
flowchart TB
  %% ── Tier 0: ground truth ──
  subgraph T0["TIER 0 — ground truth (leaves)"]
    VM["tryte-vm ref/ternary.js<br/>audited balanced-ternary arithmetic"]
    SPARK["DGX Spark<br/>GPU + training stack"]
    SURV["grokking survey<br/>(phase transitions, Fourier circuits,<br/>lazy→rich, SLT)"]
    ORIG["TSH/PDH prototypes<br/>+ Rust ports (founding artifacts)"]
  end

  %% ── Tier 1: verified primitives ──
  subgraph T1["TIER 1 — verified primitives"]
    PORT["proto/ternary.py<br/>44/44 VM vectors · 32/32 vs JS"]
    LAB["grok lab stack<br/>data / model / telemetry / train"]
    DISC["pre-registration + controls<br/>(both programs' shared law)"]
  end

  %% ── Tier 2: constructions + measured facts ──
  subgraph T2H["TIER 2A — hash family"]
    FAMV1["family v1 (T1–T4)"]
    VEC["frozen vectors + cross-language<br/>truth harnesses (22/22, 40/40)"]
    FAMV2["family v2/v3 revisions"]
    FIND1["measured findings:<br/>padding collision · class invariance ·<br/>dead pipe · 3-adic kernel · tick replay"]
    STGATE["state-level diffusion gate<br/>(born from expander-masking finding)"]
  end
  subgraph T2G["TIER 2B — grokking findings"]
    GFIND["F1–F17 (~565 runs)"]
    F17["F17: quantization forces the<br/>concentrated-Fourier pathway"]
    F13F15["F13/F15: plateau = frozen partial<br/>Fourier basin, ~2% rate"]
    F2F12["F2/F12: ternary speedup exists only<br/>in the hard, wd-gated regime"]
  end

  %% ── Tier 3: attack court + selection ──
  subgraph T3["TIER 3 — attack court & selection"]
    COURT["differential / slide / multicollision<br/>batteries (budget-labeled)"]
    MERGE["merge decision<br/>T4-v2 standard · T1/T2 alternates"]
    PUB["publish/ bundle for outside attack"]
  end

  %% ── Tier 4: bridge experiments ──
  subgraph T4X["TIER 4 — bridge experiments"]
    E1["E1 known-break distinguisher<br/>T1-v1 broken control vs T1-v2<br/>→ neural round threshold R*"]
    E2["E2 ternary-native tasks<br/>bt-add-3^k · gf729-mul · zp733-mul<br/>→ structure vs representation"]
    E3["E3 grok-as-gate<br/>learnability of r-round cores"]
    E4["E4 plateau anatomy<br/>subgroup-closed frozen support?"]
  end

  %% ── Tier 5: downstream value ──
  subgraph T5["TIER 5 — downstream"]
    OUTATT["outside attack results<br/>(the only court that creates value)"]
    NGATE["neural-distinguisher gate<br/>adopted into timeline law"]
    THEORY["learnability ≠ differentialability<br/>(grokking × cryptanalysis theory)"]
    MATCH["matched-precision ternary result<br/>(ternary nets × ternary algebra)"]
    NATIVE["native Rust re-bench<br/>(seq 85 open prediction)"]
    TICK["tick-bound fix (531,441)<br/>family iteration v4"]
  end

  %% edges: tier 0 → 1
  VM --> PORT
  SPARK --> LAB
  SURV --> LAB
  ORIG --> VEC

  %% tier 1 → 2
  PORT --> FAMV1
  PORT -.->|instrument law| FAMV1
  FAMV1 --> FIND1
  FAMV1 --> VEC
  FIND1 --> FAMV2
  FAMV2 --> VEC
  FIND1 --> STGATE
  LAB --> GFIND
  GFIND --> F17
  GFIND --> F13F15
  GFIND --> F2F12
  DISC -.-> FIND1
  DISC -.-> GFIND

  %% tier 2 → 3
  FAMV2 --> COURT
  STGATE --> COURT
  VEC --> COURT
  COURT --> MERGE
  MERGE --> PUB
  VEC --> PUB

  %% tier 3/2 → 4  (the bridges)
  FAMV2 -->|"known-broken specimens<br/>(T1-v1, T2-v1 kept in record)"| E1
  LAB --> E1
  VEC -->|"classical transition baseline<br/>(T1: round 4/18)"| E1
  PORT -->|"exact balanced-ternary &<br/>GF(3^6) reference algebra"| E2
  LAB --> E2
  F17 -->|"motivates matched-precision Q"| E2
  E1 -->|"instrument must pass<br/>broken control first"| E3
  COURT -->|"reduced-round variant loader<br/>(already built)"| E3
  E2 --> E4
  F13F15 --> E4

  %% tier 4 → 5
  E1 -->|"calibration R* vs 4/18"| NGATE
  E3 --> NGATE
  E1 --> THEORY
  E3 --> THEORY
  E2 --> MATCH
  E4 -->|"representation-theory account<br/>of the plateau"| THEORY
  PUB --> OUTATT
  VEC --> NATIVE
  FIND1 -->|"replay bound 729²"| TICK
```

## Reading the graph

**The spine** (leftmost path): audited arithmetic → verified port →
family → findings → revisions → court → merge → publish → outside
attack. This is the TSH-512 timeline's build order, complete through
seq 89.

**The load-bearing dependency nobody planned**: the state-level
diffusion gate exists *because* the dead-pipe finding proved digest
statistics are blind behind an expander. Gates in this program are
themselves discoveries, not design choices — that's why `STGATE` sits
downstream of `FIND1`, not beside it.

**The two bridges into Tier 4** are deliberately asymmetric:

- **E1** consumes the hash program's *specimens* (labeled broken
  controls — the thing neural-cryptanalysis papers never have) and the
  grok program's *instrument*. Its one hard dependency: the classical
  transition baseline (T1 round 4/18) so R\* has something to
  calibrate against. E1 gates E3 — never trust a learnability probe
  whose control hasn't separated.
- **E2** consumes the hash program's *algebra* (the exact GF(3⁶) /
  balanced-ternary reference that took a full phase to verify) and the
  grok program's *question* (F17: does quantization determine where
  solutions live — and does matched precision change the answer). E2
  gates E4.

**Critical path to new knowledge**: E1 tonight (everything it needs
exists today), then E3 in parallel with E2; E4 waits on E2's task
families. The longest path to *value* remains
`PUB → OUTATT` — no experiment here substitutes for outside attack.

**Orphan nodes (nothing depends on them yet)**: `NATIVE` (the seq-85
prediction that the Python economy ranking won't transfer to Rust) and
`TICK` (the 531,441 tick bound) — both are cheap, both are owed before
any deployment story, neither blocks the experiments.

## Node inventory (what exists vs what's proposed)

| Node | Status | Evidence |
|---|---|---|
| VM / PORT | done | timeline seq 22–23 |
| FAMV1 / FIND1 / FAMV2 | done | seq 26–47, 53 |
| STGATE | done | seq 51 pre-registration, 54/65/74 use |
| COURT / MERGE / PUB | done | seq 37–82, 87–89 |
| GFIND (F1–F17) | done | lab RESULTS.md, ~565 runs |
| E1–E4 | **proposed** | this document |
| NGATE / THEORY / MATCH | blocked on E1–E3 / E2 / E4 | — |
| OUTATT | open (outside world) | publish/ is live |
| NATIVE / TICK | open (owed) | seq 85, seq 53 |
