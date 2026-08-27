#!/usr/bin/env python3
"""verify_family_rust.py — rank-1 TRUTH for the v2 family: Rust port
(rust/verify_family.rs driving rust/trit_family_v2.rs) vs the frozen
fixture vectors/trit_family_vectors_v2.json, on every vector.

Protocol: "<alg> <tick> <input-hex>" in feed order, answers in feed
order; digest compared as the 486-char trit string (the fixture's own
serialization). Any mismatch = port WRONG (canon law), reported with
the first differing tryte.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'rust' / 'target' / 'verify_family.exe'
if sys.platform != 'win32':
    EXE = EXE.with_suffix('')
VEC = ROOT / 'vectors' / 'trit_family_vectors_v2.json'


def first_diff_trit(want: str, got: str) -> int:
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            return i
    return -1


def main() -> int:
    fixture = json.loads(VEC.read_text('utf-8'))
    lines = []
    expect = {}
    for row in fixture['vectors']:
        for tag in ('T1', 'T2', 'T3', 'T4'):
            alg = tag.lower()
            lines.append('%s %d %s' % (alg, row['tick'], row['input_hex']))
            expect[(alg, row['input_hex'], row['tick'])] = row[tag]

    proc = subprocess.run([str(EXE)], input='\n'.join(lines) + '\n',
                          capture_output=True, text=True, check=True)
    answers = proc.stdout.split('\n')
    ok = 0
    bad = 0
    for req, ans in zip(lines, answers):
        if not ans.strip():
            continue
        alg, tick, inhex = req.split(' ')
        got_alg, digest = ans.split(' ')
        assert got_alg == alg, 'protocol desync'
        want = expect[(alg, inhex, int(tick))]
        if digest == want:
            ok += 1
        else:
            bad += 1
            d = first_diff_trit(want, digest)
            print('MISMATCH %s tick=%s input=%s: first differing trit '
                  'at %d (of 486)' % (alg, tick, inhex[:16] or '(empty)', d))
    print('rust vs python (v2 family): %d exact, %d mismatch'
          % (ok, bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
