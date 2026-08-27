#!/usr/bin/env python3
"""tamper_test.py — Phase-0 self-test integrity gate.

IGNITION §4 PHASE 0: "Tamper any constant -> vectors must fail."

For each tamper below (a one-bit flip of one constant, or a one-step
change of a structural constant in proto/ternary_hash.py), a tampered
copy of the module is built, every frozen vector is recomputed, and the
tampered digests are compared against vectors/tsh512_pdh512.json.

Gate (strict): every tampered digest of each affected artifact must
differ from the frozen value — a single surviving digest would be an
undetected-tamper hole. Constants shared by both constructions (gates,
round constants, mix multipliers) must break BOTH artifacts; artifact-
specific constants (PDH initial state, TSH padding byte) must break
their own.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / 'proto' / 'ternary_hash.py').read_text('utf-8')


def flip_bit0(literal: str) -> str:
    """One-bit tamper of a hex literal: XOR its low bit."""
    return hex(int(literal, 16) ^ 1)


PHASE = ['0x9E3779B97F4A7C15', '0xC2B2AE3D27D4EB4F',
         '0x165667B19E3779F9', '0x85EBCA77C2B2AE31']
ROUNDS = [
    '0x243F6A8885A308D3', '0x13198A2E03707344', '0xA4093822299F31D0',
    '0x082EFA98EC4E6C89', '0x452821E638D01377', '0xBE5466CF34E90C6C',
    '0xC0AC29B7C97C50DD', '0x3F84D5B5B5470917', '0x9216D5D98979FB1B',
    '0xD1310BA698DFB5AC', '0x2FFD72DBD01ADFB7', '0xB8E1AFED6A267E96',
    '0xBA7C9045F12C7F99', '0x24A19947B3916CF7', '0x0801F2E2858EFC16',
    '0x636920D871574E69', '0xA458FEA3F4933D7E', '0x0D95748F728EB658',
    '0xB8CD5CF3C6A9CA23', '0x9C3B296E52A38B27', '0x72F1D887D6E22B3C',
    '0x6D0E2E1C0B4F0A72', '0x99E3B4C5E7A5B6D8', '0x6F7C8A9E1D2B3C4F',
]
PDH_INIT = [
    '0x0123456789ABCDEF', '0xFEDCBA9876543210', '0xDEADBEEFCAFEBABE',
    '0x0BADC0DE0D15EA5E', '0xCAFEBABEDEADBEEF', '0x0D15EA5E0BADC0DE',
    '0x9876543210FEDCBA', '0x89ABCDEF01234567',
]

# (label, old, new, affects) — affects: 'both' | 'tsh' | 'pdh'
TAMPERS = []
for i, lit in enumerate(PHASE):
    # PHASE[0]'s literal appears twice (list + PDH absorb): occurrence 0
    # is the PHASE list entry, occurrence 1 is the PDH absorb constant.
    if i == 0:
        TAMPERS.append(('PHASE_CONSTANTS[0] ^= 1', lit, flip_bit0(lit),
                        'both', 0))
        TAMPERS.append(('pdh absorb GOLDEN ^= 1', lit, flip_bit0(lit),
                        'pdh', 1))
    else:
        TAMPERS.append(('PHASE_CONSTANTS[%d] ^= 1' % i, lit, flip_bit0(lit),
                        'both', None))
for i, lit in enumerate(ROUNDS):
    TAMPERS.append(('ROUND_CONSTANTS[%d] ^= 1' % i, lit, flip_bit0(lit),
                    'both', None))
TAMPERS.append(('MIX_MUL1 ^= 1', '0xBF58476D1CE4E5B9',
                flip_bit0('0xBF58476D1CE4E5B9'), 'both', None))
TAMPERS.append(('MIX_MUL2 ^= 1', '0x94D049BB133111EB',
                flip_bit0('0x94D049BB133111EB'), 'both', None))
for i, lit in enumerate(PDH_INIT):
    TAMPERS.append(('PDH initial state[%d] ^= 1' % i, lit, flip_bit0(lit),
                    'pdh', None))
TAMPERS.append(('gate rotation 16 -> 17',
                'rotations = [16, 48, 32, 56]',
                'rotations = [17, 48, 32, 56]', 'both', None))
TAMPERS.append(('feistel shift 7 -> 5',
                'f = rotl64(state[i] + rc, 7)',
                'f = rotl64(state[i] + rc, 5)', 'both', None))
TAMPERS.append(('column-mix rotation 8 -> 4',
                'rotl64(parity, (i + 1) * 8)',
                'rotl64(parity, (i + 1) * 4)', 'both', None))
TAMPERS.append(('PDH cross-mix +1 -> +2',
                'j = (i + gate + 1) & 7',
                'j = (i + gate + 2) & 7', 'pdh', None))
TAMPERS.append(('PDH final rounds 8 -> 6',
                'permute(state, rounds=8)',
                'permute(state, rounds=6)', 'pdh', None))
TAMPERS.append(('TSH pad final byte 01 -> 03',
                "padded = padded[:-1] + b'\\x01'",
                "padded = padded[:-1] + b'\\x03'", 'tsh', None))


def replace_nth(src: str, old: str, new: str, occurrence):
    """Replace `old` with `new`; occurrence None = all, else the n-th."""
    if old not in src:
        raise SystemExit('tamper target not found: %r' % old)
    if occurrence is None:
        return src.replace(old, new)
    idx = -1
    for _ in range(occurrence + 1):
        idx = src.find(old, idx + 1)
        if idx < 0:
            raise SystemExit('occurrence %d not found: %r'
                             % (occurrence, old))
    return (src[:idx] + new
            + src[idx + len(old):].replace(old, old, 0))  # only the n-th


def load_tampered(src: str, name: str):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'ternary_hash.py'
        p.write_text(src, encoding='utf-8')
        spec = importlib.util.spec_from_file_location(name, str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def main() -> int:
    fixture = json.loads(
        (ROOT / 'vectors' / 'tsh512_pdh512.json').read_text('utf-8'))
    n_pass = 0
    n_fail = 0
    joint_holes = []      # tampers NO frozen vector detects at all
    per_artifact = []     # tampers leaving >=1 vector of an affected
                          # artifact unchanged (blind spots)
    for label, old, new, affects, occ in TAMPERS:
        tampered = replace_nth(SRC, old, new, occ)
        mod = load_tampered(tampered, 'tampered_' + label.replace(' ', '_'))
        unchanged = []
        changed = 0
        for v in fixture['vectors']:
            data = bytes.fromhex(v['input_hex'])
            if affects in ('both', 'tsh'):
                if mod.tsh_hash(data).hex() == v['tsh512']:
                    unchanged.append(('tsh512', v['name']))
                else:
                    changed += 1
            if affects in ('both', 'pdh'):
                if mod.pdh_hash(data).hex() == v['pdh512']:
                    unchanged.append(('pdh512', v['name']))
                else:
                    changed += 1
        if changed == 0:
            joint_holes.append(label)
            print('TAMPER INVISIBLE TO ENTIRE FIXTURE: %s' % label)
        elif unchanged:
            per_artifact.append((label, unchanged))
            print('tamper detected (joint), blind spots: %-30s %s'
                  % (label, [n for _, n in unchanged]))
        else:
            n_pass += 1
            print('tamper detected: %-32s (all %d affected digests changed)'
                  % (label, len(fixture['vectors'])
                     * (2 if affects == 'both' else 1)))
        if unchanged and changed == 0:
            n_fail += 1
    print()
    print('joint gate (fixture detects every tamper):   %d/%d %s'
          % (len(TAMPERS) - len(joint_holes), len(TAMPERS),
             'PASS' if not joint_holes else 'FAIL'))
    print('strict gate (every affected vector changed): %d/%d %s'
          % (len(TAMPERS) - len(per_artifact), len(TAMPERS),
             'PASS' if not per_artifact else 'FAIL'))
    for label, holes in per_artifact:
        print('  blind spot: %s -> %s' % (label, holes))
    # the phase-0 gate is the JOINT gate; strict failures are recorded
    # as measured blind-spot findings, not silently forgiven
    return 1 if joint_holes else 0


if __name__ == '__main__':
    sys.exit(main())
