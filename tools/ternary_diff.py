#!/usr/bin/env python3
"""ternary_diff.py — canon-law differential: proto/ternary.py (Python
port) vs ref/ternary.js (the audited reference), via node.

Runs tools/ternary_digest.js (node) and recomputes the identical digest
streams from the Python port. Domains are exhaustive: all 729 unary
values, all 531,441 binary pairs, tgate over every (control, x) at five
fixed y. A single behavioral difference flips a digest.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'proto'))
import ternary as T  # noqa: E402

ALL = list(range(T.TRYTE_MIN, T.TRYTE_MAX + 1))


class Digest:
    def __init__(self):
        self._h = hashlib.sha256()

    def update(self, s):
        self._h.update((str(s) + '\n').encode('ascii'))

    def hex(self):
        return self._h.hexdigest()


def build_streams():
    out = {}

    def unary(name, fn, arg=None):
        d = Digest()
        for v in ALL:
            d.update(fn(v, arg) if arg is not None else fn(v))
        out['unary/%s%s' % (name, '' if arg is None else '/%d' % arg)] = \
            d.hex()

    def binary(name, fn):
        d = Digest()
        for a in ALL:
            for b in ALL:
                d.update(fn(a, b))
        out['binary/%s' % name] = d.hex()

    unary('tritString', T.trit_string)
    unary('toTrits', lambda v: ','.join(str(t) for t in T.to_trits(v)))
    unary('roundtrip', lambda v: T.from_trits(T.to_trits(v)))
    unary('neg', T.neg)
    unary('sign', T.sign)
    unary('lowTrit', T.low_trit)
    unary('wrap', T.wrap)
    for n in (1, 2, 3):
        unary('shiftLeft', T.shift_left, n)
        unary('shiftRight', T.shift_right, n)
    unary('parseTritString(tritString(v))',
          lambda v: T.parse_trit_string(T.trit_string(v)))

    binary('add', T.add)
    binary('sub', T.sub)
    binary('mul', T.mul)
    binary('tMIN', T.t_min)
    binary('tMAX', T.t_max)
    binary('tCLAMPSUM', T.t_clampsum)

    tables = {
        'MIN': T.tgate_mode_table(0),
        'NEG': T.tgate_mode_table(-1),
        'MAX': T.tgate_mode_table(1),
        'ID': T.identity_table(),
        'ADDT': T.add_carry_table(),
    }
    for name, tbl in tables.items():
        binary('ttable/%s' % name,
               lambda a, b, tbl=tbl: T.ttable(tbl, a, b))
    binary('ttableC/ADDT', lambda a, b: T.ttable_c(tables['ADDT'], a, b))

    for y in (-364, -45, 0, 91, 364):
        d = Digest()
        for c in ALL:
            for x in ALL:
                d.update(T.tgate(c, x, y))
        out['tgate/y=%d' % y] = d.hex()

    # NB: must serialize in the JS object's INSERTION order (JSON.stringify
    # preserves it), so sort_keys stays off and the dict is built to mirror
    # tools/ternary_digest.js field-for-field.
    vectors = json.dumps({
        'fold2': T.fold_tables(tables['MIN'], tables['NEG']),
        'fold4': T.fold_tables(tables['MIN'], tables['NEG'],
                               tables['MAX'], tables['NEG']),
        # JS digest lists exactly these four keys (no ADDT) — mirror it
        'keys': {'MIN': T.table_key(tables['MIN']),
                 'NEG': T.table_key(tables['NEG']),
                 'MAX': T.table_key(tables['MAX']),
                 'ID': T.table_key(tables['ID'])},
        'identity': tables['ID'],
        'addt': tables['ADDT'],
        'tritString': [T.trit_string(v)
                       for v in (-364, -1, 0, 1, 2, 5, 123, 364)],
    }, separators=(',', ':'))
    out['vectors'] = hashlib.sha256(
        vectors.encode('utf-8')).hexdigest()
    return out


def main() -> int:
    js = subprocess.run(
        ['node', str(ROOT / 'tools' / 'ternary_digest.js')],
        capture_output=True, text=True, check=True)
    js_streams = json.loads(js.stdout)
    py_streams = build_streams()
    assert set(js_streams) == set(py_streams), 'stream name sets differ'
    bad = [k for k in sorted(js_streams) if js_streams[k] != py_streams[k]]
    for k in bad:
        print('DIGEST MISMATCH: %s\n  js %s\n  py %s'
              % (k, js_streams[k][:16], py_streams[k][:16]))
    print('cross-language differential: %d/%d streams match '
          '(domains: 729 unary, 531,441 pairs, tgate 729x729 x 5 y)'
          % (len(js_streams) - len(bad), len(js_streams)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
