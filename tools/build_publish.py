#!/usr/bin/env python3
"""build_publish.py — assemble the Phase-5 publication bundle (IGNITION
§4: publish vectors and constructions for OUTSIDE attack).

The bundle is GENERATED, not hand-assembled: the attack ledger is
derived from timeline.jsonl itself (the chain is the ledger), and every
file's sha256 is sealed into publish/MANIFEST.json (canonical JSON).

Layout:
  publish/README.md          challenge document + claims header
  publish/MANIFEST.json      sha256 of every bundled file
  publish/timeline.jsonl     the complete append-only record (83+ records)
  publish/ATTACK_LEDGER.md   curated attack records, generated from the chain
  publish/vectors/...        frozen fixtures (family v2 + TSH/PDH founding)
  publish/constructions/...  Python reference (+ its ternary.py), Rust core,
                             Rust verify harness
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'publish'

FILES = [
    ('vectors/trit_family_vectors_v2.json',
     ROOT / 'vectors' / 'trit_family_vectors_v2.json'),
    ('vectors/tsh512_pdh512.json',
     ROOT / 'vectors' / 'tsh512_pdh512.json'),
    ('constructions/ternary.py', ROOT / 'proto' / 'ternary.py'),
    ('constructions/trit_family_v2.py',
     ROOT / 'proto' / 'trit_family_v2.py'),
    ('constructions/trit_family_v2.rs',
     ROOT / 'rust' / 'trit_family_v2.rs'),
    ('constructions/verify_family.rs',
     ROOT / 'rust' / 'verify_family.rs'),
]

# subjects whose records form the curated attack ledger (the FULL chain
# ships as timeline.jsonl regardless)
LEDGER_SUBJECTS = (
    'attack', 'proto/tsh-padding', 'proto/pdh-tamper-blindspots',
    't1/class-invariance', 't2/dead-pipe', 't3/3adic-kernel',
    'family/tick-replay', 'family/prefix-birthday',
    't1/padding-injectivity', 'truth/v2-family-rust',
)

README = """# THE TRIT HASH FAMILY — publication for outside attack

*Published from the TSH-512 timeline (branch of cruise118/aethor,
chronicle seq 3302). Law: IGNITION.md — obscurity is not security;
every resistance claim below is budget-labeled; unattacked = UNKNOWN.*

## What this is

A custom balanced-ternary hash family, built std-only on the tryte-vm's
audited arithmetic. After a pre-registered gate contest and internal
attack court, **T4-v2 (TRIT-MD) is the family standard**; T1-v2
(sponge) and T2-v3 (ARX) are published alternates. All three:

- pass all five pre-registered gates (determinism; avalanche in the
  trit band [0.6, 0.73]; measured diffusion; tamper; state-level
  diffusion),
- agree cross-language bit-for-bit: Python reference vs Rust core,
  40/40 frozen vectors,
- survived the internal attack budgets listed in ATTACK_LEDGER.md.

## Claims header — read before attacking

The strongest claim this timeline makes: **"survived attack budget X;
unknown beyond."** Everything here is a research artifact. It is NOT a
drop-in replacement for blake2b/sha3.

**Published bounds and known weaknesses (stated before you find them):**

1. **Tick domain: UNBOUNDED as of family v4** (timeline seq 100). The
   tick is encoded in minimal balanced-ternary digits — no modular
   field. Historical: v3 replayed exactly at distance 729² = 531,441
   (measured, seq 53); v4 closed it structurally (measured to 10⁹+,
   seq 100). Message domain stays <= 531,440 trytes.
2. **T4's output expander** chains `c = wrap(c*7 + w[j%16] + seed)` —
   we expect state-word leakage from digests to be the first fruitful
   outside attack (pre-registered at timeline seq 84).
3. **T4 state diffusion** has 1/96 chaining trits at 0.573 change rate
   (below the 0.6 band center, above the 0.5 floor) at full 3 passes.
4. **Economy ranking** (why T4 leads) is Python-prototype scale:
   T4 0.123 > T2 0.0032 > T1 0.0008 MB/s — instrument-labeled, native
   ratios will differ (pre-registered at seq 85).
5. **Founding artifacts TSH-512/PDH-512 are published WITH their
   breaks**: TSH has a free padding collision `TSH(X) == TSH(X||0x80)`
   for `len(X) ≡ 30 (mod 32)` (in the fixture as a known-collision
   pair); PDH has 16 dead round constants and 0x00-absorb-neutrality.
   They are the record's starting point, not sound hashes.

## Verify (std-only; no build deps beyond rustc/python/node)

    # cross-language truth (40/40):
    rustc --edition 2021 -O constructions/verify_family.rs -o verify_family
    python3 tools/../verify_family_rust.py   # or feed lines yourself:
    #   echo "t4 0 616263" | ./verify_family
    # Python reference determinism against the frozen vectors:
    python3 -c "import sys; sys.path.insert(0,'constructions'); \\
        import trit_family_v2 as F, json; \\
        vecs=json.load(open('vectors/trit_family_vectors_v2.json'))['vectors']; \\
        print(all(F.output_string(getattr(F,{'T1':'t1_hash','T2':'t2_hash','T3':'t3_hash','T4':'t4_hash'}[t])(bytes.fromhex(r['input_hex']),r['tick']))==r[t] for r in vecs for t in ('T1','T2','T3','T4')))"

The arithmetic base (`constructions/ternary.py`) is a test-for-test
port of the tryte-vm's audited `ref/ternary.js`: 44/44 of the VM's own
test vectors, 32/32 exhaustive differential streams vs the JS.

## What counts as a break

Any pair `H(x) == H(y)` with `x != y` constructible faster than
birthday; any recovery of internal state from digests; any differential
characteristic with probability materially above random at reduced
rounds; any violation of the published bounds above. Findings belong on
the timeline (append-only, hash-chained `timeline.jsonl` is included).

*Curl's lesson is founding law: value forms only where attackers fail.
Attack this.*
"""


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_ledger():
    lines = ['# ATTACK LEDGER — curated from timeline.jsonl',
             '',
             'The complete append-only, hash-chained record ships as '
             '`timeline.jsonl`.',
             'Below: every attack/finding record this timeline '
             'pre-registered or measured.', '']
    n = 0
    with open(ROOT / 'timeline.jsonl', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            subj = rec.get('subject', '')
            if rec.get('kind') in ('finding', 'prediction') and \
                    subj.startswith(LEDGER_SUBJECTS):
                n += 1
                lines.append('## seq %d — %s [%s] %s' % (
                    rec['seq'], rec.get('kind'), subj,
                    rec.get('detail', '')[:400]
                    + ('…' if len(rec.get('detail', '')) > 400 else '')))
                lines.append('')
    lines.insert(3, '(%d records shown)' % n)
    return '\n'.join(lines) + '\n'


def main():
    if PUB.exists():
        shutil.rmtree(PUB)
    for sub in ('vectors', 'constructions'):
        (PUB / sub).mkdir(parents=True)
    copied = []
    for rel, src in FILES:
        dst = PUB / rel
        shutil.copyfile(src, dst)
        copied.append(rel)
    shutil.copyfile(ROOT / 'timeline.jsonl', PUB / 'timeline.jsonl')
    copied.append('timeline.jsonl')
    (PUB / 'ATTACK_LEDGER.md').write_text(build_ledger(), 'utf-8')
    copied.append('ATTACK_LEDGER.md')
    (PUB / 'README.md').write_text(README, 'utf-8')
    copied.append('README.md')
    manifest = {'v': 1, 'files': {rel: sha256(PUB / rel) for rel in copied}}
    (PUB / 'MANIFEST.json').write_text(
        canon(manifest) + '\n', 'utf-8', newline='\n')
    print('bundle: %d files -> %s (MANIFEST.json sealed)'
          % (len(copied), PUB))


if __name__ == '__main__':
    main()
