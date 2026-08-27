#!/usr/bin/env python3
"""freeze_vectors.py — freeze Phase-0 vectors for TSH-512 / PDH-512.

IGNITION §4 PHASE 0: publish fixed vectors (empty, 1 byte, "abc",
32/64/128-byte, the tick encodings 0/1/2), verify Python <-> Rust
bit-exactness on every vector (tools/verify_rust.py drives
rust/verify_vectors.rs over this same file), and tamper-test every
constant (tools/tamper_test.py recomputes against this file).

The known TSH padding collision (timeline seq 13: TSH(X) == TSH(X||0x80)
for len(X) mod 32 == 30) is sealed INTO the fixture as a known-collision
pair — the defect is now part of the frozen record, not an anecdote.

One serialization (canon law): sorted keys, no whitespace, UTF-8.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'proto'))
from ternary_hash import tsh_hash, pdh_hash  # noqa: E402

VECTORS = [
    ('empty', b''),
    ('one-byte-a5', b'\xa5'),
    ('abc', b'abc'),
    ('len32', bytes(range(32))),
    ('len64', bytes(range(64))),
    ('len128', bytes(range(128))),
    # tick encodings: tick index as its minimal unsigned byte
    # (TSH/PDH as imported have no in-construction tick mixing; Phase 0
    # freezes their behavior on the tick ENCODINGS as inputs)
    ('tick0', b'\x00'),
    ('tick1', b'\x01'),
    ('tick2', b'\x02'),
]

DETERMINISM_REPS = 100


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def main() -> int:
    rows = []
    for name, data in VECTORS:
        # determinism gate: DETERMINISM_REPS recomputes, byte-identical
        tsh_reps = {tsh_hash(data) for _ in range(DETERMINISM_REPS)}
        pdh_reps = {pdh_hash(data) for _ in range(DETERMINISM_REPS)}
        if len(tsh_reps) != 1 or len(pdh_reps) != 1:
            print('NON-DETERMINISTIC: %s' % name)
            return 1
        rows.append({
            'name': name,
            'input_hex': data.hex(),
            'tsh512': tsh_reps.pop().hex(),
            'pdh512': pdh_reps.pop().hex(),
        })

    # known-collision fixture (timeline seq 13): len(X) mod 32 == 30
    kc_a = bytes(range(30))
    kc_b = kc_a + b'\x80'
    assert tsh_hash(kc_a) == tsh_hash(kc_b)
    assert pdh_hash(kc_a) != pdh_hash(kc_b)

    doc = {
        'v': 1,
        'law': 'IGNITION.md PHASE 0 (timeline branch of aethor seq 3302)',
        'reference': 'proto/ternary_hash.py',
        'rust': 'rust/tsh.rs, rust/pdh.rs (driven by rust/verify_vectors.rs)',
        'determinism_reps': DETERMINISM_REPS,
        'tick_encoding': 'tick index as minimal unsigned byte '
                         '(00 / 01 / 02)',
        'vectors': rows,
        'known_collision_tsh512': {
            'a_input_hex': kc_a.hex(),
            'b_input_hex': kc_b.hex(),
            'shared_tsh512': tsh_hash(kc_a).hex(),
            'pdh512_a': pdh_hash(kc_a).hex(),
            'pdh512_b': pdh_hash(kc_b).hex(),
            'cause': '10*1 padding overwrites the 0x80 marker when it '
                     'lands on a block-final byte; timeline seq 13',
        },
    }
    out = ROOT / 'vectors' / 'tsh512_pdh512.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(canon(doc) + '\n')
    print('froze %d vectors (%d digests) + 1 known-collision pair -> %s'
          % (len(rows), len(rows) * 2, out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
