#!/usr/bin/env python3
"""ternary_port_tests.py — the tryte-vm's OWN test vectors, run against
proto/ternary.py (the Python port of ref/ternary.js).

IGNITION PHASE 1: "The port must pass the VM's own test vectors before
anything builds on it."

Sections 0–6 are test/test-core.js translated verbatim (same exhaustive
domains: all 729 trytes, all 531,441 pairs). Section 7 ports the
test-fold.js essentials (fold exactness, closure, hash-consing, TOPC).
Section 8 adds this timeline's OWN additions beyond the VM's suite: the
independent schoolbook-multiplication oracle (the VM's arithmetic
oracle is circular — it compares wrap against wrap) and an EXHAUSTIVE
(rather than stride-5) ADD-vs-TOPC check.
"""
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'proto'))
import ternary as T  # noqa: E402

PASS = 0
FAILS = []


def check(name, cond, detail=''):
    global PASS
    if cond:
        PASS += 1
        print('  [PASS] %s' % name)
    else:
        FAILS.append(name)
        print('  [FAIL] %s  %s' % (name, detail))


ALL = list(range(T.TRYTE_MIN, T.TRYTE_MAX + 1))

print('\n=== 0. INSTRUMENT SELF-TEST ===')
check('tritString(0) is six zeros', T.trit_string(0) == '000000')
check('tritString(1) ends in +', T.trit_string(1) == '00000+')
check('5 is 9-3-1  (000+--)', T.trit_string(5) == '000+--')
check('364 is all +', T.trit_string(364) == '++++++')
# prove the harness detects a false claim
_probe_ok = False
try:
    assert 1 + 1 == 3
except AssertionError:
    _probe_ok = True
check('harness detects a false claim', _probe_ok)

print('\n=== 1. REPRESENTATION (all 729 values) ===')
check('trit round-trip is lossless',
      all(T.from_trits(T.to_trits(v)) == v for v in ALL))
check('string round-trip is lossless',
      all(T.parse_trit_string(T.trit_string(v)) == v for v in ALL))
check('all 729 encodings are distinct',
      len({T.trit_string(v) for v in ALL}) == 729)
check('every trit is in {-1,0,+1}',
      all(t in (-1, 0, 1) for v in ALL for t in T.to_trits(v)))

print('\n=== 2. ARITHMETIC vs ORACLE (531,441 pairs per op) ===')
bad = None
for a in ALL:
    for b in ALL:
        if T.add(a, b) != T.wrap(a + b):
            bad = ('add', a, b)
            break
        if T.sub(a, b) != T.wrap(a - b):
            bad = ('sub', a, b)
            break
        if T.mul(a, b) != T.wrap(a * b):
            bad = ('mul', a, b)
            break
    if bad:
        break
check('add / sub / mul over all %d tryte pairs' % (729 * 729), bad is None,
      repr(bad))
check('every result stays in range',
      all(T.TRYTE_MIN <= T.add(a, b) <= T.TRYTE_MAX
          for a in ALL for b in ALL))

print('\n=== 3. THE FOUR PROPERTIES THAT MOTIVATE BALANCED TERNARY ===')
check('(a) negation IS flipping every trit',
      all(T.neg(v) == T.from_trits([-t for t in T.to_trits(v)])
          for v in ALL))
check('(a) ...and equals arithmetic negation',
      all(T.neg(v) == -v for v in ALL))
check('(b) no sign bit: -v is the exact mirror of v',
      all(T.trit_string(-v) == T.trit_string(v)
          .replace('+', '#').replace('-', '+').replace('#', '-')
          for v in ALL))
check('(c) sign = sign of leading non-zero trit',
      all(T.sign(v) == (v > 0) - (v < 0) for v in ALL))
check('(d) truncation rounds to nearest, ties away from zero',
      all(T.shift_right(v, 1) == (1 if v > 0 else -1 if v < 0 else 0)
          * ((abs(v) + 1) // 3) for v in ALL))
check('subtraction needs no separate circuit: a-b === a+neg(b)',
      all(T.sub(a, b) == T.add(a, T.neg(b))
          for a in ALL for b in ALL))

print('\n=== 4. SHIFTS ===')
check('shiftLeft(v,n) === v * 3^n (wrapped)',
      all(T.shift_left(v, n) == T.wrap(v * 3 ** n)
          for v in ALL for n in (0, 1, 2, 3)))
check('shiftRight undoes shiftLeft when nothing overflowed',
      all(T.shift_right(T.shift_left(v, 2), 2) == v
          for v in ALL if abs(v) <= 40))

print('\n=== 5. TRITWISE GATES (531,441 pairs each) ===')


def per_trit(fn, a, b):
    tb = T.to_trits(b)
    return T.from_trits([fn(t, tb[i])
                         for i, t in enumerate(T.to_trits(a))])


g = True
for a in ALL:
    for b in ALL:
        if T.t_min(a, b) != per_trit(min, a, b):
            g = False
            break
        if T.t_max(a, b) != per_trit(max, a, b):
            g = False
            break
        if T.t_clampsum(a, b) != per_trit(
                lambda p, q: max(-1, min(1, p + q)), a, b):
            g = False
            break
    if not g:
        break
check('MIN / MAX / CLAMPSUM are exactly tritwise', g)
check('MIN and MAX are commutative',
      all(T.t_min(a, b) == T.t_min(b, a)
          and T.t_max(a, b) == T.t_max(b, a)
          for a in ALL for b in ALL))
ROWS = {(-1, -1): -1, (-1, 0): -1, (-1, 1): 0,
        (0, -1): -1, (0, 0): 0, (0, 1): 1,
        (1, -1): 0, (1, 0): 1, (1, 1): 1}
check('CLAMPSUM reproduces the 9-row table from the source conversation',
      all(T.to_trits(T.t_clampsum(T.from_trits([p, 0, 0, 0, 0, 0]),
                                  T.from_trits([q, 0, 0, 0, 0, 0])))[0] == o
          for (p, q), o in ROWS.items()))

print('\n=== 6. TGATE — the reconfigurable unit ===')
ALL_NEG = T.parse_trit_string('------')
ALL_ZERO = T.parse_trit_string('000000')
ALL_POS = T.parse_trit_string('++++++')
check('control = all 0  collapses TGATE to MIN',
      all(T.tgate(ALL_ZERO, a, b) == T.t_min(a, b)
          for a in ALL for b in ALL))
check('control = all +  collapses TGATE to MAX',
      all(T.tgate(ALL_POS, a, b) == T.t_max(a, b)
          for a in ALL for b in ALL))
check('control = all -  collapses TGATE to NEG',
      all(T.tgate(ALL_NEG, a, 0) == T.neg(a) for a in ALL))
def tgate_per_position_ok(c, x=123, y=-45):
    modes = T.to_trits(c)
    xs = T.to_trits(x)
    ys = T.to_trits(y)
    got = T.to_trits(T.tgate(c, x, y))
    return all(got[i] == (-xs[i] if modes[i] == -1 else
                          min(xs[i], ys[i]) if modes[i] == 0 else
                          max(xs[i], ys[i]))
               for i in range(6))


check('each of the 729 controls names a circuit; per-trit selection '
      'is honoured',
      all(tgate_per_position_ok(c) for c in ALL))
check('the 729 controls collapse to fewer than 729 distinct functions '
      'on fixed data',
      len({T.tgate(c, 123, -45) for c in ALL}) < 729)

print('\n=== 7. FOLD / TABLES / TOPC (test-fold.js essentials) ===')
MODES = [T.tgate_mode_table(0), T.tgate_mode_table(-1),
         T.tgate_mode_table(1), T.tgate_mode_table(-1)]

f2 = T.fold_tables(MODES[0], MODES[1])
check('a two-stage fold matches the chain for every tryte, at several y',
      all(T.ttable(f2, x, y) == T.ttable(MODES[1], T.ttable(MODES[0], x, y), y)
          for y in (-364, -45, 0, 91, 364) for x in ALL))
f4 = T.fold_tables(*MODES)


def chain4(x, y):
    v = x
    for t in MODES:
        v = T.ttable(t, v, y)
    return v


check('a four-stage fold matches the chain for every tryte, at several y',
      all(T.ttable(f4, x, y) == chain4(x, y)
          for y in (-364, -13, 0, 123, 364) for x in ALL))
check('the fold formula is y-independent — one fold serves every second '
      'operand',
      all(T.ttable(f2, x, y) == T.ttable(MODES[1], T.ttable(MODES[0], x, y), y)
          for y in (-364, -100, 0, 55, 364) for x in ALL))
check('the identity table folds away without trace',
      T.table_key(T.fold_tables(T.identity_table(), MODES[2]))
      == T.table_key(MODES[2])
      and T.table_key(T.fold_tables(MODES[2], T.identity_table()))
      == T.table_key(MODES[2]))
check('folding is associative, as composition must be',
      T.table_key(T.fold_tables(T.fold_tables(MODES[0], MODES[1]), MODES[2]))
      == T.table_key(T.fold_tables(MODES[0], T.fold_tables(MODES[1], MODES[2]))))


def rnd(s):
    return [((s * 7 + i * 13) % 3) - 1 for i in range(9)]


check('every fold of two TOP tables is again a TOP table '
      '(checked on 400 pairs)',
      all(len(T.fold_tables(rnd(a), rnd(b))) == 9
          and all(-1 <= e <= 1 for e in T.fold_tables(rnd(a), rnd(b)))
          for a in range(20) for b in range(20)))
modes = {T.table_key(T.tgate_mode_table(m)) for m in (-1, 0, 1)}
comps = {T.table_key(T.fold_tables(T.tgate_mode_table(a), T.tgate_mode_table(b)))
         for a in (-1, 0, 1) for b in (-1, 0, 1)}
check("but 6 of TGATE's 8 two-stage composites are not any TGATE mode",
      len(comps) == 8
      and len([c for c in comps if c not in modes]) == 6)
check('identical tables share one content address',
      T.table_key(MODES[0]) == T.table_key(T.tgate_mode_table(0))
      and T.table_key(MODES[0]) != T.table_key(MODES[1]))
check('entries differing only above the lowest trit address identically',
      T.table_key(MODES[0]) == T.table_key(
          [e + (3 if i % 2 else -3) for i, e in enumerate(MODES[0])]))
check('S3 — four tables with two distinct values need 2 x 9 cells',
      len({T.table_key(t) for t in [MODES[0], MODES[1], MODES[0], MODES[1]]})
      * 9 == 18)

ADDT = T.add_carry_table()
check('the table has 27 rows, one per (x, y, carry-in)', len(ADDT) == 27)
check('each row packs both outputs into one cell: out + 3 x carry',
      all(-4 <= v <= 4 for v in ADDT))
check('ADD equals TOPC with the addition table, over every operand pair',
      all(T.ttable_c(ADDT, x, y) == T.add(x, y)
          for x in ALL for y in ALL))
check('U1 — no TOP table equals ADD, over all 19,683, compared as '
      'functions',
      not any(all(T.ttable(
          [(n // 3 ** k) % 3 - 1 for k in range(9)], x, y) == T.add(x, y)
          for x, y in [(i * 53 - 364, i * 137 - 364) for i in range(40)])
          for n in range(19683)))
check('   ...though 81 of them match ADD at one input pair, which is '
      'the trap',
      sum(1 for n in range(19683)
          if T.ttable([(n // 3 ** k) % 3 - 1 for k in range(9)], 123, -45)
          == T.add(123, -45)) == 81)

print('\n=== 8. THIS TIMELINE\'S ADDITIONS (beyond the VM suite) ===')


def mul_oracle(a, b):
    """Schoolbook balanced-trit multiplication on raw trit arrays.

    Independent of T.mul (which is wrap(a*b)): the only shared code is
    the representation itself. 12-trit accumulator, wrapped once at the
    end — the circularity the VM's own oracle has is removed.
    """
    at = T.to_trits(a)
    bt = T.to_trits(b)
    acc = [0] * 12
    for i, x in enumerate(at):
        carry = 0
        for j, y in enumerate(bt):
            s = acc[i + j] + x * y + carry
            r = s % 3
            if r == 2:
                r = -1
            acc[i + j] = r
            carry = (s - r) // 3
        acc[i + 6] += carry
    return T.wrap(T.from_trits(acc))


check('independent schoolbook mul oracle agrees on all 531,441 pairs',
      all(T.mul(a, b) == mul_oracle(a, b) for a in ALL for b in ALL))

# encoding sanity for the family to come: balanced ternary of the tick
check('tick encodings 0/1/2 as single trits are 0/+/- ... 2 = +- '
      '(one up, one down)',
      T.trit_string(0) == '000000' and T.trit_string(1) == '00000+'
      and T.trit_string(2) == '0000+-')

print('\n' + '=' * 64)
print('FAILED (%d): %s' % (len(FAILS), ', '.join(FAILS)) if FAILS
      else 'ALL %d CHECKS PASSED' % PASS)
print('=' * 64)
sys.exit(1 if FAILS else 0)
