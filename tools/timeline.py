#!/usr/bin/env python3
"""timeline.py — the TSH-512 timeline's strict history.

Append-only, hash-chained record of every act on this timeline:
predictions, results, findings, failures, decrees. Same canon law as
the parent Aethor chronicle this timeline branched from.

  - APPEND-ONLY: never edit or delete a record; a correction is a NEW
    record naming the record it corrects.
  - CANONICAL: hash covers canonical JSON (sorted keys, no whitespace)
    of every field except `hash` itself, chained to `prev` (the prior
    record's hash; GENESIS is 64 zeros).
  - VERIFY: recompute the whole chain; the FIRST mismatch is printed
    with its line — that is the finding, never a silent skip.

Usage:
  python tools/timeline.py record --kind prediction --actor <you> \
      --subject tsh/avalanche --detail "predict: bit avalanche drifts high"
  python tools/timeline.py verify
  python tools/timeline.py head
  python tools/timeline.py query --kind failure
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'timeline.jsonl'
LOCK = LEDGER.with_suffix('.lock')
GENESIS = '0' * 64
V = 1


def canon(obj) -> str:
    """The one serialization: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)


def _acquire(timeout: float = 10.0) -> None:
    """Cross-process append lock — the chain must never fork."""
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode('ascii'))
            os.close(fd)
            return
        except FileExistsError:
            if time.time() > deadline:
                raise SystemExit('timeline: ledger lock busy '
                                 '(another writer is active)')
            time.sleep(0.05)


def _release() -> None:
    try:
        LOCK.unlink()
    except OSError:
        pass


def _records():
    if not LEDGER.exists():
        return
    with open(LEDGER, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != 'hash'}
    return hashlib.sha256(canon(body).encode('utf-8')).hexdigest()


def append(kind: str, actor: str, subject: str, detail: str,
           payload_path: str = '') -> None:
    recs = list(_records())
    rec = {
        'v': V, 't': round(time.time(), 3),
        'seq': (recs[-1]['seq'] + 1) if recs else 1,
        'kind': kind, 'actor': actor, 'subject': subject,
        'detail': detail, 'payload_sha256': '', 'prev': GENESIS,
    }
    if recs:
        rec['prev'] = recs[-1]['hash']
    if payload_path:
        p = Path(payload_path)
        rec['payload_sha256'] = hashlib.sha256(
            p.read_bytes()).hexdigest()
    rec['hash'] = record_hash(rec)
    _acquire()
    try:
        # re-read under lock: another writer may have advanced the head
        latest = list(_records())
        if latest:
            if latest[-1]['hash'] != rec['prev']:
                rec['prev'] = latest[-1]['hash']
                rec['seq'] = latest[-1]['seq'] + 1
                rec['hash'] = record_hash(rec)
        with tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                         dir=str(ROOT), delete=False,
                                         newline='\n') as tmp:
            tmp.write(canon(rec) + '\n')
            tmp.flush()
            os.fsync(tmp.fileno())
        # append under lock (the temp+append keeps a partial write from
        # ever landing mid-line)
        with open(LEDGER, 'a', encoding='utf-8', newline='\n') as f:
            with open(tmp.name, 'r', encoding='utf-8') as src:
                f.write(src.read())
            f.flush()
            os.fsync(f.fileno())
        os.unlink(tmp.name)
    finally:
        _release()
    print(json.dumps({'seq': rec['seq'], 'hash': rec['hash'][:16]}))


def verify() -> int:
    prev = GENESIS
    findings = 0
    for i, rec in enumerate(_records(), start=1):
        if rec.get('prev') != prev:
            print(f'CHAIN BREAK at line {i}: prev {rec.get("prev")[:16]}'
                  f' != expected {prev[:16]}')
            findings += 1
        if rec.get('seq') != i:
            print(f'SEQ MISMATCH at line {i}: {rec.get("seq")} != {i}')
            findings += 1
        expected = record_hash(rec)
        if rec.get('hash') != expected:
            print(f'HASH MISMATCH at line {i}: recorded '
                  f'{rec.get("hash", "")[:16]} != computed '
                  f'{expected[:16]}')
            findings += 1
        if findings:
            break
        prev = rec['hash']
    if not findings:
        n = i if LEDGER.exists() else 0
        print(f'timeline: {n} records, chain intact, head '
              f'{prev[:16]}')
    return 1 if findings else 0


def head() -> None:
    recs = list(_records())
    if not recs:
        print('timeline: empty')
        return
    last = recs[-1]
    print(json.dumps({'seq': last['seq'], 'hash': last['hash'][:16],
                      'kind': last['kind']}))


def query(kind: str) -> None:
    n = 0
    for rec in _records():
        if rec.get('kind') == kind:
            print(canon(rec))
            n += 1
    if not n:
        print(f'(no {kind} records)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record')
    r.add_argument('--kind', required=True)
    r.add_argument('--actor', required=True)
    r.add_argument('--subject', required=True)
    r.add_argument('--detail', required=True)
    r.add_argument('--payload', default='',
                   help='path to a file whose sha256 seals into the record')
    sub.add_parser('verify')
    sub.add_parser('head')
    q = sub.add_parser('query')
    q.add_argument('--kind', required=True)
    args = ap.parse_args()
    if args.cmd == 'record':
        append(args.kind, args.actor, args.subject, args.detail,
               args.payload)
    elif args.cmd == 'verify':
        sys.exit(verify())
    elif args.cmd == 'head':
        head()
    else:
        query(args.kind)


if __name__ == '__main__':
    main()
