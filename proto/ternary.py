#!/usr/bin/env python3
"""ternary.py — Python port of ref/ternary.js (the tryte-vm's audited
balanced-ternary core), test-for-test per IGNITION PHASE 1.

Law of the port: match ref/ternary.js behavior exactly — same LSB-first
trit arrays, same wrap() semantics, same table layouts (x-major), same
packed carry cells. This module is std-only and is the arithmetic base
the T1–T4 family (SPEC.md) must build on. Verified by:
  tools/ternary_port_tests.py  — the VM's own test vectors (test-core.js
                                 + test-fold.js essentials), translated
  tools/ternary_diff.py        — exhaustive differential dump vs the
                                 actual JS via node (canon law)
"""
import math

TRITS_PER_TRYTE = 6
RADIX = 3
TRYTE_STATES = RADIX ** TRITS_PER_TRYTE      # 729
TRYTE_MAX = (TRYTE_STATES - 1) // 2          # +364
TRYTE_MIN = -TRYTE_MAX                       # -364


def wrap(v):
    """Fold any integer into the balanced tryte range, wrapping mod 729.

    Python's % is already non-negative for a positive modulus, so the
    JS double-mod idiom collapses to one %.
    """
    return (v - TRYTE_MIN) % TRYTE_STATES + TRYTE_MIN


def to_trits(v, n=TRITS_PER_TRYTE):
    """Integer -> list of n trits, least-significant first.

    Out-of-range input wraps, exactly like the reference. Python's % is
    non-negative, matching the reference's ((x % 3) + 3) % 3.
    """
    x = wrap(v)
    out = []
    for _ in range(n):
        r = x % RADIX                # 0, 1, 2
        if r == 2:
            r = -1                   # 2 carries up and becomes -1
        out.append(r)
        x = (x - r) // RADIX
    return out


def from_trits(trits):
    """List of trits (LSB first) -> integer."""
    v = 0
    for t in reversed(trits):
        v = v * RADIX + t
    return v


def trit_string(v):
    """Human-readable form, most-significant first: '+' / '0' / '-'."""
    return ''.join('+' if t > 0 else '-' if t < 0 else '0'
                   for t in reversed(to_trits(v)))


def parse_trit_string(s):
    """Parse '+-0++0' (MSB first) back to an integer."""
    m = {'+': 1, '-': -1, '0': 0}
    trits = []
    for c in s.strip():
        if c not in m:
            raise ValueError('bad trit character: %s' % c)
        trits.append(m[c])
    return from_trits(list(reversed(trits)))


# --- arithmetic -------------------------------------------------------------

def add(a, b):
    return wrap(a + b)


def sub(a, b):
    return wrap(a - b)


def mul(a, b):
    return wrap(a * b)


def neg(a):
    """Negation: flip every trit. Provably identical to arithmetic negation."""
    return from_trits([-t for t in to_trits(a)])


def sign(a):
    """Sign is the sign of the leading (most significant) non-zero trit."""
    for t in reversed(to_trits(a)):
        if t != 0:
            return t
    return 0


def shift_left(a, n):
    """Shift left n trits — multiply by 3^n, wrapping."""
    return wrap(a * RADIX ** n)


def shift_right(a, n):
    """Shift right n trits.

    Dropping low trits in balanced ternary rounds to nearest (ties away
    from zero), which is why no rounding logic is needed.
    """
    t = to_trits(a)
    kept = t[n:] + [0] * n
    return from_trits(kept)


# --- tritwise gates ---------------------------------------------------------

def _tritwise2(fn):
    def op(a, b):
        x = to_trits(a)
        y = to_trits(b)
        return from_trits([fn(t, y[i]) for i, t in enumerate(x)])
    return op


def _clamp(p, q):
    return max(-1, min(1, p + q))


t_min = _tritwise2(min)
t_max = _tritwise2(max)
t_clampsum = _tritwise2(_clamp)

t_neg = neg


def tgate(control, x, y):
    """TGATE — a control tryte reconfigures the unit per trit position.

      control trit -1  ->  NEG(x)        (ignores y)
      control trit  0  ->  MIN(x, y)
      control trit +1  ->  MAX(x, y)
    """
    c = to_trits(control)
    xs = to_trits(x)
    ys = to_trits(y)
    out = []
    for i, mode in enumerate(c):
        if mode == -1:
            out.append(-xs[i])
        elif mode == 0:
            out.append(min(xs[i], ys[i]))
        else:
            out.append(max(xs[i], ys[i]))
    return from_trits(out)


def low_trit(v):
    """The least significant trit of a value. Exact because shift_right
    rounds to nearest."""
    w = wrap(v)
    return w - 3 * shift_right(w, 1)


def ttable(table, x, y):
    """Apply an arbitrary two-input ternary truth table, tritwise.

    Row order x-major: index = (x + 1) * 3 + (y + 1). Only the LOWEST
    TRIT of each entry is used — total for any table values.
    """
    xs = to_trits(x)
    ys = to_trits(y)
    return from_trits([low_trit(table[(xt + 1) * 3 + (ys[i] + 1)])
                       for i, xt in enumerate(xs)])


def tgate_mode_table(mode):
    """The nine-entry table equivalent to one TGATE mode, for a fixed
    y-trit."""
    out = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            out.append(-x if mode == -1
                       else min(x, y) if mode == 0
                       else max(x, y))
    return out


def fold_tables(*tables):
    """Fold a chain of TOP tables into ONE table that does the same job.

        T3[a][b] = T2[ T1[a][b] ][b]

    Column-wise composition; y is not mentioned, so a chain folds once
    at build time and stays correct for every input it will ever see.
    The fold stops at a carry: ADD couples positions, so it is not
    positionwise and no single table can stand for it.
    """
    def fold_two(t1, t2):
        out = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                mid = low_trit(t1[(a + 1) * 3 + (b + 1)])
                out.append(low_trit(t2[(mid + 1) * 3 + (b + 1)]))
        return out

    acc = tables[0]
    for t in tables[1:]:
        acc = fold_two(acc, t)
    return acc


def table_key(table):
    """Content address for a table (lowest-trit canonical form)."""
    return ','.join(str(low_trit(e)) for e in table)


def identity_table():
    """The identity table: passes x through untouched, whatever y is."""
    return [x for x in (-1, 0, 1) for _ in (-1, 0, 1)]


def ttable_c(table, x, y):
    """TTABLEC — a table-driven operation WITH A CARRY.

    27 rows, x-major: index = (x + 1) * 9 + (y + 1) * 3 + (c + 1). Each
    row holds both outputs packed as out + 3 * carry. Positions are
    processed least significant first, carry starts at 0, and the final
    carry is dropped — ADD's wrapping semantics.
    """
    xs = to_trits(x)
    ys = to_trits(y)
    out = []
    c = 0
    for i in range(TRITS_PER_TRYTE):
        cell = wrap(table[(xs[i] + 1) * 9 + (ys[i] + 1) * 3 + (c + 1)])
        out.append(low_trit(cell))
        c = shift_right(cell, 1)
    return from_trits(out)


def add_carry_table():
    """The 27-row table for balanced ternary addition. ADD is a table.

    The JS formula (sum - 3*carry) + 3*carry reduces to sum — the cell
    IS the raw sum in [-3, 3]; low_trit extracts the out trit and
    shift_right(cell, 1) the carry. Ported literally (kept as the same
    expression) so the reduction stays visible.
    """
    t = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for c in (-1, 0, 1):
                s = x + y + c
                # JS: Math.sign(sum) * Math.round(Math.abs(sum) / 3)
                # |sum| in [0,3] -> round-half-up on 0, 1/3, 2/3, 1
                carry = int(math.copysign(math.floor(abs(s) / 3 + 0.5), s)) \
                    if s else 0
                t.append((s - 3 * carry) + 3 * carry)
    return t
