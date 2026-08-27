#!/usr/bin/env node
'use strict';
/* ternary_digest.js — canonical differential digest of ref/ternary.js.
 *
 * One side of the canon-law cross-language fixture (IGNITION §2 law 4):
 * this script and tools/ternary_diff.py compute THE SAME digest streams
 * over THE SAME domains. Any behavioral difference between the JS
 * reference and the Python port must surface as a digest mismatch.
 *
 * Streams (all in canonical order; every item serialized as
 * String(value) + '\n'):
 *   unary/<fn>/<arg>   for every v in [-364, 364]
 *   binary/<fn>        for every (a, b) pair, a outer, b inner
 *   ttable/<name>      for every (a, b) pair, named table
 *   ttableC/ADDT       for every (a, b) pair
 *   tgate/y=<y>        for every (control, x), control outer
 *   vectors            exact JSON of tables (fold/mode/key/identity)
 */
const crypto = require('crypto');
const T = require('../ref/ternary.js');

const ALL = [];
for (let v = T.TRYTE_MIN; v <= T.TRYTE_MAX; v++) ALL.push(v);

function digest() {
  const h = crypto.createHash('sha256');
  return {
    update(s) { h.update(s + '\n'); },
    hex() { return h.digest('hex'); },
  };
}

const out = {};

function unary(name, fn, arg) {
  const d = digest();
  for (const v of ALL) d.update(String(fn(v, arg)));
  out['unary/' + name + (arg === undefined ? '' : '/' + arg)] = d.hex();
}

function binary(name, fn) {
  const d = digest();
  for (const a of ALL) for (const b of ALL) d.update(String(fn(a, b)));
  out['binary/' + name] = d.hex();
}

unary('tritString', T.tritString);
unary('toTrits', v => T.toTrits(v).join(','));
unary('roundtrip', v => T.fromTrits(T.toTrits(v)));
unary('neg', T.neg);
unary('sign', T.sign);
unary('lowTrit', T.lowTrit);
unary('wrap', T.wrap);
for (const n of [1, 2, 3]) {
  unary('shiftLeft', T.shiftLeft, n);
  unary('shiftRight', T.shiftRight, n);
}
unary('parseTritString(tritString(v))', v => T.parseTritString(T.tritString(v)));

binary('add', T.add);
binary('sub', T.sub);
binary('mul', T.mul);
binary('tMIN', T.tMIN);
binary('tMAX', T.tMAX);
binary('tCLAMPSUM', T.tCLAMPSUM);

const TABLES = {
  MIN: T.tgateModeTable(0),
  NEG: T.tgateModeTable(-1),
  MAX: T.tgateModeTable(1),
  ID: T.identityTable(),
  ADDT: T.addCarryTable(),
};
for (const [name, tbl] of Object.entries(TABLES)) {
  binary('ttable/' + name, (a, b) => T.ttable(tbl, a, b));
}
binary('ttableC/ADDT', (a, b) => T.ttableC(TABLES.ADDT, a, b));

for (const y of [-364, -45, 0, 91, 364]) {
  const d = digest();
  for (const c of ALL) for (const x of ALL) d.update(String(T.tgate(c, x, y)));
  out['tgate/y=' + y] = d.hex();
}

out['vectors'] = digest_of(JSON.stringify({
  fold2: T.foldTables(TABLES.MIN, TABLES.NEG),
  fold4: T.foldTables(TABLES.MIN, TABLES.NEG, TABLES.MAX, TABLES.NEG),
  keys: {
    MIN: T.tableKey(TABLES.MIN),
    NEG: T.tableKey(TABLES.NEG),
    MAX: T.tableKey(TABLES.MAX),
    ID: T.tableKey(TABLES.ID),
  },
  identity: TABLES.ID,
  addt: TABLES.ADDT,
  tritString: [-364, -1, 0, 1, 2, 5, 123, 364].map(T.tritString),
}));

function digest_of(s) {
  return crypto.createHash('sha256').update(s).digest('hex');
}

console.log(JSON.stringify(out));
