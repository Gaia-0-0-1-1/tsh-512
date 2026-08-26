'use strict';
/* ---------------------------------------------------------------------------
 * ternary.js — balanced ternary core
 *
 * A TRIT is one of -1, 0, +1.
 * A TRYTE is 6 trits. 3^6 = 729 distinct values, held balanced: -364 .. +364.
 *
 * Balanced ternary has no sign bit. The sign of a number is the sign of its
 * leading non-zero trit, and negation is simply flipping every trit. There is
 * no two's complement, no separate subtract circuit, and truncating a value
 * rounds it to nearest rather than toward zero. Those four properties are the
 * whole reason this number system is interesting, and each is asserted in
 * test/test.js rather than merely claimed here.
 * ------------------------------------------------------------------------- */

const TRITS_PER_TRYTE = 6;
const RADIX = 3;
const TRYTE_STATES = RADIX ** TRITS_PER_TRYTE;   // 729
const TRYTE_MAX = (TRYTE_STATES - 1) / 2;        // +364
const TRYTE_MIN = -TRYTE_MAX;                    // -364

/** Fold any integer into the balanced tryte range, wrapping mod 729. */
function wrap(v) {
  return ((((v - TRYTE_MIN) % TRYTE_STATES) + TRYTE_STATES) % TRYTE_STATES) + TRYTE_MIN;
}

/** Integer -> array of 6 trits, least-significant first. Wraps out-of-range input. */
function toTrits(v, n = TRITS_PER_TRYTE) {
  let x = wrap(v);
  const out = [];
  for (let i = 0; i < n; i++) {
    let r = ((x % RADIX) + RADIX) % RADIX;   // 0, 1, 2
    if (r === 2) r = -1;                     // 2 carries up and becomes -1
    out.push(r);
    x = (x - r) / RADIX;
  }
  return out;
}

/** Array of trits (LSB first) -> integer. */
function fromTrits(trits) {
  let v = 0;
  for (let i = trits.length - 1; i >= 0; i--) v = v * RADIX + trits[i];
  return v;
}

/** Human-readable form, most-significant first: '+' / '0' / '-'. */
function tritString(v) {
  return toTrits(v).reverse().map(t => (t > 0 ? '+' : t < 0 ? '-' : '0')).join('');
}

/** Parse '+-0++0' (MSB first) back to an integer. */
function parseTritString(s) {
  const trits = s.trim().split('').map(c => {
    if (c === '+') return 1;
    if (c === '-') return -1;
    if (c === '0') return 0;
    throw new Error(`bad trit character: ${c}`);
  });
  return fromTrits(trits.reverse());
}

/* --- arithmetic ---------------------------------------------------------- */

const add = (a, b) => wrap(a + b);
const sub = (a, b) => wrap(a - b);
const mul = (a, b) => wrap(a * b);

/** Negation: flip every trit. Provably identical to arithmetic negation. */
function neg(a) {
  return fromTrits(toTrits(a).map(t => -t));
}

/** Sign is the sign of the leading (most significant) non-zero trit. */
function sign(a) {
  const t = toTrits(a);
  for (let i = t.length - 1; i >= 0; i--) if (t[i] !== 0) return t[i];
  return 0;
}

/** Shift left n trits — multiply by 3^n, wrapping. */
const shiftLeft = (a, n) => wrap(a * RADIX ** n);

/**
 * Shift right n trits. Dropping low trits in balanced ternary rounds to
 * nearest (ties away from zero), which is why no rounding logic is needed.
 */
function shiftRight(a, n) {
  const t = toTrits(a);
  const kept = t.slice(n).concat(new Array(n).fill(0));
  return fromTrits(kept);
}

/* --- tritwise gates ------------------------------------------------------ */
/* The ternary-native analogs. MIN generalises AND, MAX generalises OR, and
 * NEG generalises NOT. CLAMPSUM is the 9-row gate published in the source
 * conversation, confirmed there as the saturating ternary sum. */

const tritwise2 = fn => (a, b) => {
  const x = toTrits(a), y = toTrits(b);
  return fromTrits(x.map((t, i) => fn(t, y[i])));
};

const tMIN = tritwise2((p, q) => Math.min(p, q));
const tMAX = tritwise2((p, q) => Math.max(p, q));
const tCLAMPSUM = tritwise2((p, q) => Math.max(-1, Math.min(1, p + q)));
const tNEG = neg;

/**
 * TGATE — the idea the source conversation was reaching for, in ternary-native
 * form. A control tryte reconfigures the unit per trit position: each control
 * trit selects which gate acts on that slice of the two data trytes.
 *
 *   control trit -1  ->  NEG(x)        (ignores y)
 *   control trit  0  ->  MIN(x, y)
 *   control trit +1  ->  MAX(x, y)
 *
 * One tryte of control therefore names one of 3^6 = 729 distinct 6-slice
 * circuits, and the data reconfigures the unit as it arrives.
 */
function tgate(control, x, y) {
  const c = toTrits(control), xs = toTrits(x), ys = toTrits(y);
  return fromTrits(c.map((mode, i) => {
    if (mode === -1) return -xs[i];
    if (mode === 0) return Math.min(xs[i], ys[i]);
    return Math.max(xs[i], ys[i]);
  }));
}

/**
 * TTABLE — apply an arbitrary two-input ternary truth table, tritwise.
 *
 * A two-input trit function is nine rows, so nine entries fully specify one of
 * the 3^9 = 19,683 such functions. Row order is x-major:
 *
 *      index = (x + 1) * 3 + (y + 1)      x, y in {-1, 0, +1}
 *
 * Only the LOWEST TRIT of each entry is used, which is this machine's own idiom
 * for reading a trit out of a tryte (v - 3*(v>>1)) and makes the operation total:
 * no table can be malformed enough to fault.
 *
 * TGATE is the special case where the table is chosen per position from three
 * fixed candidates; this chooses one table for all positions but allows any of
 * the 19,683.
 */
function ttable(table, x, y) {
  const xs = toTrits(x), ys = toTrits(y);
  return fromTrits(xs.map((xt, i) => lowTrit(table[(xt + 1) * 3 + (ys[i] + 1)])));
}

/** The least significant trit of a value. Exact because shiftRight rounds. */
function lowTrit(v) {
  const w = wrap(v);
  return w - 3 * shiftRight(w, 1);
}

/** The nine-entry table equivalent to one TGATE mode, for a fixed y-trit. */
function tgateModeTable(mode) {
  const out = [];
  for (let x = -1; x <= 1; x++) {
    for (let y = -1; y <= 1; y++) {
      out.push(mode === -1 ? -x : mode === 0 ? Math.min(x, y) : Math.max(x, y));
    }
  }
  return out;
}

/**
 * Fold a chain of TOP tables into ONE table that does the same job.
 *
 *      T3[a][b] = T2[ T1[a][b] ][b]
 *
 * Column-wise composition, and notice what is missing: y. The formula does not
 * mention the second operand, so a chain folds once at build time and stays
 * correct for every input it will ever see.
 *
 * This works because TOP's columns are arbitrary, so the family is CLOSED under
 * composition. TGATE's are not: its three modes generate a seven-element per-trit
 * monoid, so most composites of two TGATEs are not any TGATE. Being foldable is a
 * property TOP has and TGATE does not.
 *
 * The fold stops at a carry. ADD couples trit positions, so a chain broken by one
 * is not positionwise any more and no single table can stand for it — the same
 * boundary the carry audit measured.
 */
function foldTables(...tables) {
  return tables.reduce((t1, t2) => {
    const out = [];
    for (let a = -1; a <= 1; a++) {
      for (let b = -1; b <= 1; b++) {
        const mid = lowTrit(t1[(a + 1) * 3 + (b + 1)]);
        out.push(lowTrit(t2[(mid + 1) * 3 + (b + 1)]));
      }
    }
    return out;
  });
}

/**
 * TTABLEC — a table-driven operation WITH A CARRY, so one instruction can reach
 * past the positionwise ceiling.
 *
 * Every table-driven operation so far computes out_p from (x_p, y_p) alone, which
 * caps the whole family at 27^w however many tables it owns. This one takes a
 * third input, the carry out of the position below, and emits a carry to the
 * position above. That is the cross-trit coupling the carry audit identified as
 * the only escape.
 *
 * 27 rows, indexed x-major:
 *
 *      index = (x + 1) * 9 + (y + 1) * 3 + (c + 1)
 *
 * Each row is ONE cell holding both outputs packed as  out + 3 * carry, which is
 * just a two-trit number: the low trit is the output, the next is the carry. The
 * machine's own arithmetic does the unpacking, so no new encoding is invented.
 *
 * Positions are processed least significant first, carry starts at 0, and the
 * final carry is dropped — exactly ADD's wrapping semantics, which is why ADD
 * turns out to be one particular table.
 */
function ttableC(table, x, y) {
  const xs = toTrits(x), ys = toTrits(y);
  const out = [];
  let c = 0;
  for (let i = 0; i < TRITS_PER_TRYTE; i++) {
    const cell = wrap(table[(xs[i] + 1) * 9 + (ys[i] + 1) * 3 + (c + 1)]);
    out.push(lowTrit(cell));
    c = shiftRight(cell, 1);
  }
  return fromTrits(out);
}

/** The 27-row table for balanced ternary addition. ADD is a table. */
function addCarryTable() {
  const t = [];
  for (let x = -1; x <= 1; x++) {
    for (let y = -1; y <= 1; y++) {
      for (let c = -1; c <= 1; c++) {
        const sum = x + y + c;
        const carry = Math.sign(sum) * Math.round(Math.abs(sum) / 3);
        t.push((sum - 3 * carry) + 3 * carry);      // out in the low trit, carry in the next
      }
    }
  }
  return t;
}

/** Content address for a table, so identical ones can share one copy of the cells. */
function tableKey(table) {
  return table.map(lowTrit).join(',');
}

/** The identity table: passes x through untouched, whatever y is. */
function identityTable() {
  const out = [];
  for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) out.push(x);
  return out;
}

module.exports = {
  TRITS_PER_TRYTE, RADIX, TRYTE_STATES, TRYTE_MAX, TRYTE_MIN,
  wrap, toTrits, fromTrits, tritString, parseTritString,
  add, sub, mul, neg, sign, shiftLeft, shiftRight,
  tMIN, tMAX, tCLAMPSUM, tNEG, tgate, ttable, lowTrit, tgateModeTable,
  foldTables, tableKey, identityTable, ttableC, addCarryTable,
};
