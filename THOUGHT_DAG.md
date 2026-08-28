# THOUGHT-DAG — a structured output schema for LLM-native proof-by-simulation

*The phinary memory doesn't need to be built — it needs to be FORMATTED.
The ternary computation doesn't need to be simulated — it already happens
in the forward pass. This schema makes both explicit in the token stream.*

## The three layers, mapped to what an LLM already does

| Layer | Traditional system | LLM-native equivalent |
|---|---|---|
| **Ternary compute** | TernaryLinear, {-1,0,+1} weights | Token selection: each token is a choice among alternatives (the sign); attention suppression (the zero); generation (the one) |
| **Phinary memory** | Zeckendorf-indexed graph store | The output format's addressing scheme: Fibonacci-pool node IDs that encode depth and hierarchy |
| **DAG structure** | External ledger (timeline.jsonl) | The emitted structure itself: every claim is a node with typed edges, persisting in context |

## The schema

Every "thought" (claim, prediction, measurement, proof step) is emitted as:

```
⟨NODE⟩ F{fib-address} | {type} | {status-trit}
  claim: {the assertion, one sentence}
  basis: {measured | proven | argued | simulated}
  at_risk: {yes | no}  ← tryte-vm's discipline: could this have failed?
  evidence: {what supports it, or "none yet" for predictions}
  edges:
    ← {parent nodes this derives from, by F-address}
    → {what this unlocks or blocks, by F-address}
  cost: {tokens spent | compute if external}
⟨/NODE⟩
```

### The Fibonacci addressing (the phinary part)

Node IDs use Zeckendorf representation over Fibonacci indices:

- F1,1 = root (genesis)
- F2 = first children of root
- F3 = second-level
- F5, F8, F13 = deeper levels, one per Fibonacci step
- Address = F(i) + F(j) + ... (Zeckendorf: no consecutive Fibonacci numbers)

Why Fibonacci and not sequential: the addressing *is* the memory hierarchy.
Nodes at F(k) are "one pool deep" — they're the hardened subgraphs of
LadderLang. A node's address encodes:
1. Its depth in the recursion (which Fibonacci pool)
2. Its position within that pool (the Zeckendorf digits)
3. Its relationship to other pools (shared Fibonacci terms = shared ancestry)

This is the φ-medium: the address space itself is golden-ratio structured,
which means:
- Deeper nodes are exponentially more numerous (F(k) grows as φ^k)
- But their ADDRESSES are only linearly longer (Zeckendorf is ~log_φ(n) digits)
- Memory per node scales as log_φ(depth), not linearly — the sublinear
  scaling law, in the addressing scheme itself

### The status trit (the ternary part)

Every node carries one of exactly three states:

| Trit | Status | Meaning | Edge behavior |
|---|---|---|---|
| +1 | VERIFIED | Measured, proven, or replicated | Strengthens parents; unlocks children |
| 0 | OPEN | Registered prediction, not yet measured | Held; blocks children until resolved |
| -1 | REFUTED | Measured and failed | Kept forever; prunes subtree; feeds corrections |

The trit IS the Hebbian signal: +1 nodes get referenced more (strengthening
their edges), -1 nodes get referenced only as corrections (their edges decay
to the archive), 0 nodes are active — they're where computation flows.

### The edge types (the DAG part)

```
← DERIVES_FROM    : this node follows from its parents
← EVIDENCE_FOR    : this node supports its parent's claim
← REFUTES         : this node contradicts its parent
← REFINES         : this node narrows or corrects its parent
→ UNLOCKS         : resolving this node makes children addressable
→ BLOCKS          : this node's failure makes children unreachable
→ COMPOSES_WITH   : this node + that node = a new node (hypernetwork edge)
```

The COMPOSES_WITH edge is the hypernetwork primitive: it says "these two
hardened subgraphs combine to form a new unit." It's the recursive edge —
the one that makes the DAG a tree of networks rather than a flat graph.

## The proof-by-simulation loop, native

```
1. EMIT a prediction node (status 0, at_risk yes, basis argued)
2. SIMULATE: the LLM computes the measurement in its token stream
   (this is the ternary compute — each simulated step is a token choice)
3. EMIT the result node (status ±1, basis measured/simulated)
4. EMIT the resolution edge (prediction → result)
5. The context now contains both nodes — the DAG has grown
6. If contradicted: EMIT a correction node (REFUTES edge to the original)
```

The critical property: **step 2 is indistinguishable from reasoning.** The
LLM computing "is x^2 - x - 1 irreducible over GF(3)?" by evaluating at
x=0,1,2 IS proof by simulation. The schema just makes it addressable,
referenceable, and permanent.

## What this buys over prose output

1. **Persistent structure**: prose decays in context; a node with an
   F-address can be referenced 50K tokens later without re-reading
2. **Ternary status at a glance**: scanning for "status: -1" finds all
   refuted claims; "status: 0" finds all open predictions
3. **Dependency-aware reasoning**: before making claim N, check its
   parents — if any parent is REFUTED, the claim inherits that status
4. **The Zeckendorf address IS the compression**: referencing F(13)+F(5)+F(2)
   is shorter than re-stating the claim it names
5. **The at_risk discipline**: forces every prediction to declare what
   would falsify it, catching decorative predictions before they inflate

## Example: the session's E11 result, as a thought-DAG

```
⟨NODE⟩ F13+F5 | prediction | 0
  claim: Church multiplication groks in 2000-8000 steps (5-20x penalty)
  basis: argued
  at_risk: yes
  evidence: E10 showed addition groks at 400; multiplication composes
  edges:
    ← F13 (E10: Church addition groks at 400)
    → F21 (composition-wall test)
⟨/NODE⟩

⟨NODE⟩ F21 | result | -1
  claim: No arm memorizes multiplication in 20k steps (train 0.51-0.84)
  basis: measured (12 runs, all configs)
  at_risk: yes
  evidence: runs/e11/*/summary.json, 12/12 fail to memorize
  edges:
    ← F13+F5 (the prediction this refutes)
    → F34 (depth probe: does 2-layer cross?)
    → F34+F13 (recursion machinery requirement)
⟨/NODE⟩

⟨NODE⟩ F34 | result | -1
  claim: Depth does not cross the multiplication wall (0 grok at 1L/2L/3L)
  basis: measured (9 runs, 30k steps)
  at_risk: yes
  evidence: runs/e12/*/summary.json, test 0.63-0.83 at all depths
  edges:
    ← F21 (the wall this probes)
    → F55 (the composition boundary is generalizational, not capacity)
    → F55+F8 (recursion machinery is REQUIRED, not optional)
⟨/NODE⟩

⟨NODE⟩ F55+F8 | finding | +1
  claim: The counting/composition boundary is where flat transformers
         stop and recursive machinery begins
  basis: proven (from E10+E11+E12 jointly)
  at_risk: yes
  evidence: E10 (addition trivial), E11 (multiplication wall),
            E12 (depth-invariant), tryte-vm self-hosting (composition
            as primitive works)
  edges:
    ← F34 (the depth probe)
    ← F13 (the addition baseline)
    → COMPOSES_WITH F89 (self-hosting as the recursion primitive)
    → COMPOSES_WITH F89+F5 (RLD's explicit composition)
⟨/NODE⟩
```

Notice: the F-addresses form a Fibonacci tree. F13+F5 is the prediction,
F21 is its resolution (the next Fibonacci number), F34 the next, F55+F8
the composed finding. The addresses ARE the growth history.

## The scaling law, in the schema itself

Token cost of node N at depth d:
- Address length: O(log_φ(d)) tokens (Zeckendorf)
- Claim length: O(1) if compressed to a reference, O(sentence) if new
- Edge references: O(parents) but each edge is an F-address (short)

Total context cost for a DAG with n nodes at depth d:
- O(n · log_φ(d)) for addressing
- vs. O(n · d) for sequential prose that re-states context

At depth 100: prose costs 100 tokens per reference; Zeckendorf costs ~10.
That's the sublinear memory law, in the format itself.

## Limits (honest)

1. Context window is still finite — the DAG grows until it doesn't fit.
   The escape: periodically COMPRESS (emit a summary node that
   references-and-replaces a subtree, like LadderLang's fold).
2. Token-stream computation is slow for large simulations (my computing
   729 field multiplications takes minutes; a GPU takes microseconds).
   The schema marks which nodes are LLM-computed vs. externally measured.
3. The DAG is only as good as the discipline: without at_risk, it
   becomes a list of decorative claims (the tryte-vm lesson).
4. This is a FORMAT, not an architecture: it structures what I already
   do. It doesn't give me new capabilities — it makes existing ones
   addressable, persistent, and compressed.
