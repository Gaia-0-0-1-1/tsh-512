#!/usr/bin/env python3
"""verify_rust.py — Python <-> Rust bit-exactness on every frozen vector.

Feeds every frozen vector input (plus the known-collision pair) through
rust/target/verify_vectors.exe (rust/verify_vectors.rs driving the
rust/tsh.rs and rust/pdh.rs cores) and compares, byte for byte, against
the frozen digests. TRUTH is fitness rank 1 (IGNITION §5): one mismatch
zeroes everything below.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'rust' / 'target' / 'verify_vectors.exe'
if sys.platform != 'win32':
    EXE = EXE.with_suffix('')


def main() -> int:
    fixture = json.loads(
        (ROOT / 'vectors' / 'tsh512_pdh512.json').read_text('utf-8'))

    lines = []
    expect = {}
    for v in fixture['vectors']:
        for alg, key in (('tsh512', 'tsh512'), ('pdh512', 'pdh512')):
            lines.append('%s %s' % (alg, v['input_hex']))
            expect[(alg, v['input_hex'])] = v[key]
    kc = fixture['known_collision_tsh512']
    # the known-collision pair: both sides, both algorithms — the Rust
    # core must reproduce the collision itself, not just the vectors
    for ihex in (kc['a_input_hex'], kc['b_input_hex']):
        lines.append('tsh512 %s' % ihex)
        lines.append('pdh512 %s' % ihex)
    expect[('tsh512', kc['a_input_hex'])] = kc['shared_tsh512']
    expect[('tsh512', kc['b_input_hex'])] = kc['shared_tsh512']
    expect[('pdh512', kc['a_input_hex'])] = kc['pdh512_a']
    expect[('pdh512', kc['b_input_hex'])] = kc['pdh512_b']

    proc = subprocess.run(
        [str(EXE)], input='\n'.join(lines) + '\n',
        capture_output=True, text=True, check=True)

    # ordered pairing: answers come back in feed order
    answers = proc.stdout.split('\n')
    n_ok = 0
    n_bad = 0
    for req, ans in zip(lines, answers):
        if not ans.strip():
            continue
        alg, inhex = req.split(' ')
        got_alg, digest = ans.split(' ')
        assert got_alg == alg, 'protocol desync: %s vs %s' % (got_alg, alg)
        want = expect[(alg, inhex)]
        if digest == want:
            n_ok += 1
        else:
            n_bad += 1
            print('MISMATCH %s input=%s\n  frozen %s\n  rust   %s'
                  % (alg, inhex[:24] or '(empty)', want[:32], digest[:32]))
    rust_tsh_collision = (
        expect[('tsh512', kc['a_input_hex'])]
        == expect[('tsh512', kc['b_input_hex'])])
    print('rust vs python: %d exact, %d mismatch '
          '(known-collision pair reproduced by Rust: %s)'
          % (n_ok, n_bad, rust_tsh_collision))
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
