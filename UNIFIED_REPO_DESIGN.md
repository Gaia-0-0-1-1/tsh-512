# THE UNIFIED REPO — design document

*The five-repo convergence (seq 143) made physical. The filesystem IS the
DAG. Folders are attention routers. The U1 schema (seq 147) is the node
format. The phinary addressing is the address space. This document is the
design; the repo it describes is the build.*

---

## 1. Design principles

### P1 — The filesystem IS the DAG
Not a representation of the DAG. The actual DAG. Every folder is a node.
Every file is a leaf. Every `node.json` declares the edges the filesystem
tree cannot express (the non-tree DAG edges: COMPOSES_WITH, DERIVES_FROM,
EVIDENCE_FOR, REFUTES).

### P2 — Folders route attention
When a human or an LLM needs X, the traversal cost is the attention cost.
The structure must answer "where would X be?" before the seeker thinks
about it. Depth = specialization. Breadth = generality. The most-referenced
artifacts live shallowest.

### P3 — Fibonacci depth, not arbitrary depth
The top level is the first pool (F2: two nodes — LAW and WORK). Each level
down is the next pool. The number of children per node is not capped, but
the *intended* structure follows the golden section: a node with many
children is asking to be split; a node with one child is asking to be
merged. φ-ratio'd subtrees: when a folder's children are ~1.6×
differentiated in size, the split is natural.

### P4 — Every node carries the U1 schema
The unified claim schema from the F55 test (seq 147), made physical:
every folder has a `node.json` with identity, status, basis, at_risk,
edges, provenance. The schema gap measured at U1 (our timeline lacked
at_risk) is fixed from day one.

### P5 — Sources are cloned, modules are lifted, actives are referenced
Three treatment tiers:
- **Archived repos** (phi-highway, tribyte corpus): clone fully into
  `sources/` — history preserved, receipts sealed.
- **Load-bearing modules** (ternary.js, rld_v5, interp.tasm, the
  featurizer, StakhovLinear): lifted into the DAG structure with a
  provenance manifest recording origin repo + commit + hash.
- **Active repos** (ternary-grokking, the Spark lab): referenced by
  pointer (a `remote.json` with fetch instructions), never forked —
  they are alive and owned by other agents.

### P6 — The repo is its own first citizen
The repo's structure obeys the same laws it documents. The THOUGHT_DAG
schema describes the folder structure. The PROTOCOL governs commits to
this repo. The fitness function (truth > gates > resistance > economy)
ranks what gets built next.

---

## 2. The top-level structure

```
unified/
│
├── LAW/                          # F1 — how to work here. Read first.
│   ├── IGNITION.md               #   the founding commission (from tsh-512)
│   ├── PROTOCOL.md               #   the recursive workflow (from tsh-512)
│   ├── METHOD.md                 #   the discovery loop (from tryte-vm)
│   ├── THOUGHT_DAG.md            #   the token-stream schema (from tsh-512)
│   ├── SCHEMA.md                 #   the U1 unified claim schema, formalized
│   └── node.json                 #   this folder's DAG node
│
├── LEDGER/                       # F2 — what is true. All five ledgers.
│   ├── timeline/                 #   tsh-512's hash chain (144 records)
│   ├── findings/                 #   tryte-vm's FINDINGS + traps (193 preds)
│   ├── receipts/                 #   tribyte bundle receipts (Spark)
│   ├── store/                    #   frontier/rld proof store (44 entries)
│   ├── results/                  #   ternary-grokking RESULTS (F1-F25)
│   └── translation.json          #   the U1 field mapping, machine-readable
│
├── COMPUTE/                      # F3 — the ternary substrate
│   ├── arithmetic/               #   ternary.js + ternary.py (both verified)
│   ├── isa/                      #   the 25-instruction machine
│   ├── selfhost/                 #   interp.tasm + interp-core.tasm
│   ├── device/                   #   the switch-level layer
│   ├── lattice/                  #   StakhovLinear, phi-lattices, a3_cascade
│   └── field/                    #   GF(3^k), f36 arithmetic, the golden element
│
├── MEMORY/                       # F4 — the phinary structures
│   ├── rld/                      #   RLD v3/v4/v5 (from phi-highway)
│   ├── namespace/                #   Zeckendorf namespace (from tribyte)
│   ├── store/                    #   the tribyte evidence-packet engine
│   ├── kv/                       #   the orphaned kv_cache.rs (to be wired)
│   └── bridge/                   #   the frontier RLD Bridge (rld.bat terminal)
│
├── LEARNING/                     # F5 — what grokking measured
│   ├── lab/                      #   the grokking lab core (model, telemetry)
│   ├── experiments/              #   E-series task definitions + results
│   ├── spectrum/                 #   the learnability spectrum (E7)
│   ├── walls/                    #   the measured walls (place-value, mult, depth)
│   └── dissect/                  #   the dissection instruments (Walsh, OV, Fourier)
│
├── BRIDGE/                       # F6 — the translations between systems
│   ├── contracts/                #   ternary-native-lab E1-E5 (unrun)
│   ├── quantization/             #   phi2 results, the phinary forensics grid
│   ├── unification/              #   the F55 test artifacts (U1/U2/U3)
│   └── mappings/                 #   cross-system translation tables
│
├── CORPUS/                       # F7 — the source material, cloned
│   ├── sources/                  #   full clones of archived repos
│   │   ├── phi-highway/
│   │   ├── tribyte-v5/           #     (Ternary-Model-local-v5)
│   │   ├── golden-chain-spark/
│   │   ├── ladderlang/
│   │   └── scp/
│   ├── remotes/                  #   pointers to active repos (not forked)
│   │   ├── ternary-grokking.json #     fetch URL, HEAD, last-pulled
│   │   ├── ternary-native-lab.json
│   │   └── tryte-vm.json
│   └── MANIFEST.json             #   every source: origin, commit, sha256, license
│
└── node.json                     # the root node: the repo's own DAG entry
```

**Why this order**: LAW before WORK (you read the constitution before the
code). LEDGER before COMPUTE (truth before capability). COMPUTE before
MEMORY (the substrate before the structure). LEARNING after both (it
studies them). BRIDGE last (it connects what exists). CORPUS at the
bottom (raw material, deepest).

---

## 3. The node format (U1 schema, physical)

Every folder carries `node.json`:

```json
{
  "address": "F8+F2",
  "name": "selfhost",
  "status": "+1",
  "kind": "artifact",
  "basis": "measured",
  "at_risk": false,
  "claim": "The self-hosting interpreter: ternary VM implemented in its own ISA, verified 3-deep",
  "provenance": {
    "origin": "tryte-vm",
    "commit": "<sha>",
    "files_sha256": {"interp.tasm": "<hash>", "interp-core.tasm": "<hash>"}
  },
  "edges": {
    "derives_from": ["COMPUTE/isa"],
    "evidence_for": ["LEARNING/walls/composition"],
    "composes_with": ["MEMORY/namespace"]
  },
  "unlocks": ["BRIDGE/contracts/E4"],
  "attention_hint": "The recursion primitive — what E12 said the ladder needs"
}
```

The `attention_hint` field is the router: one sentence telling a future
seeker (human or LLM) what this node is FOR, so traversal doesn't require
reading the contents.

---

## 4. The addressing scheme

Folders get Zeckendorf addresses at creation:

```
unified/           = F1 (root)
LAW/               = F2
WORK/              = F3 (LEDGER, COMPUTE, MEMORY, LEARNING, BRIDGE, CORPUS
                        are F5, F8, F13, F21, F34, F55 — pool members)
LEDGER/timeline    = F5+F2
COMPUTE/selfhost   = F8+F2
```

The address is metadata (the human-readable path is primary), but it:
1. Gives every node a canonical, compressed reference (~10 tokens at depth
   100, per U2's honest 1.44×)
2. Encodes pool membership — nodes sharing Fibonacci terms share ancestry
3. Makes the DAG's growth history visible in the addresses themselves
   (F55 nodes are fifth-pool: deep, specialized, composed)

---

## 5. The non-tree edges (how the DAG exceeds the filesystem)

Filesystems are trees. The DAG needs cross-links. Three mechanisms:

1. **node.json edges** (primary): every node declares its typed edges.
   A crawler can reconstruct the full DAG from the node.json files alone.
2. **Symlinks** (sparse, for the strongest links): `BRIDGE/unification`
   might symlink the F55 artifacts rather than copying them. Used only
   where the filesystem router genuinely helps.
3. **The LEDGER/translation.json** (the U1 map): the machine-readable
   mapping of all five ledger schemas, so a claim in one system can be
   found in any other.

---

## 6. Migration plan (what gets cloned, lifted, referenced)

| Source | Treatment | Destination |
|---|---|---|
| tsh-512 | **lift** (LAW, LEDGER/timeline, COMPUTE/field, THOUGHT_DAG) + reference | multiple |
| tryte-vm | **lift** (COMPUTE: arithmetic/isa/selfhost/device) + reference for the rest | COMPUTE/* |
| phi-highway | **clone** (archived — the RLD mothership) | CORPUS/sources/ |
| Ternary-Model-local-v5 | **lift** (MEMORY/namespace) + clone | both |
| golden-chain-spark | **clone** (the φ/Zeckendorf math program) | CORPUS/sources/ |
| ladderlang | **clone** (Lean proofs) | CORPUS/sources/ |
| scp | **clone** (receipt engine) | CORPUS/sources/ |
| frontier/rld | **lift** (MEMORY/bridge, LEDGER/store) | both |
| ternary-grokking | **reference** (ACTIVE — another agent owns it) | CORPUS/remotes/ |
| ternary-native-lab (Spark) | **lift** (BRIDGE/contracts) + reference | both |
| Spark lab (~/workstation/lab) | **reference** (ACTIVE) | CORPUS/remotes/ |

**The rule**: lift what the DAG needs as routing nodes; clone what needs
its history preserved; reference what is alive. Nothing active gets
frozen.

---

## 7. The first build actions (ordered)

1. `git init` the unified repo; write the root `node.json`
2. Create the top-level structure with LAW/ populated first (the
   constitution files, lifted from tsh-512)
3. Write `SCHEMA.md` — the U1 schema formalized (fixing the at_risk gap)
4. Clone the archived sources into CORPUS/sources/ (phi-highway,
   golden-chain, ladderlang, scp, tribyte-v5)
5. Lift COMPUTE/ from tryte-vm (the five verified modules + node.json
   provenance for each)
6. Lift LEDGER/ (the timeline, the findings, the receipts, the store)
7. Write LEDGER/translation.json (the U1 mapping, machine-readable)
8. Lift MEMORY/ and BRIDGE/ (RLD, namespace, contracts)
9. Write the remotes for the active repos
10. First crawl: a script that walks all node.json files and reconstructs
    the full DAG, verifying every edge points to a real node

Each action is a commit. Each commit seals provenance. The repo is born
under the law it carries.

---

## 8. What this buys (honest)

1. **One attention surface**: an LLM agent (or human) dropped into the
   repo root can traverse to any finding in the five-program corpus by
   folder-routing, without knowing which original repo it lived in.
2. **The U1 gap closed**: the at_risk discipline is structural from
   day one, not retrofitted.
3. **The DAG is crawlable**: a script (or an agent) can reconstruct the
   complete dependency graph from node.json files — the RLD's
   "weights ARE memory" made physical in the filesystem.
4. **Provenance is sealed**: every lifted module records its origin
   commit and content hash. The CORPUS/MANIFEST.json is the receipt.
5. **Active work is not frozen**: the live labs stay referenced, not
   forked. The unified repo is the map; the territories stay sovereign.

## 9. What this does NOT buy (honest)

1. It does not merge the codebases — a module lifted from tryte-vm still
   expects its own dependencies. The lifts are curated references, not
   integrations.
2. It does not replace the live labs — it indexes them. New experiments
   happen in the active repos; the unified repo's LEARNING/ updates by
   reference and periodic sync.
3. The addressing scheme is metadata, not enforcement — folders can
   violate the Fibonacci pools; the addresses document intent, not law.
4. It is one more repo to maintain. The cost is real; the payoff must
   be measured (P-W1's stranger-pass-rate is the metric: does work
   organized this way get found faster?).
