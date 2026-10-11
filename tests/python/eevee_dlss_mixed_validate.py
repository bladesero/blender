# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Compare mixed RR acceptance outputs, independently of successful render/API return codes.

Run with a Python containing numpy after the eevee_dlss_compatibility.py fixtures.
The root contains final_forward300, final_reverse300, final_single150 and their .err logs.
This checks numerical/lifecycle evidence; it does not certify visual quality or GPU speed.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def frames(directory):
    return {int(path.stem.split('frame')[-1]): path for path in directory.glob('*_frame*.npy')}


def compare(a, b):
    a, b = np.load(a), np.load(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.abs(a - b)
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}


def validate(root, tolerance):
    forward = frames(root / 'final_forward300')
    reverse = frames(root / 'final_reverse300')
    assert set(forward) == set(reverse) == set(range(1, 301)), 'Missing animation frames'
    order = {str(frame): compare(forward[frame], reverse[frame]) for frame in forward}
    single = compare(forward[150], frames(root / 'final_single150')[150])
    affine = compare(root / 'final_AFFINE_packed/00_frame0001.npy',
                     root / 'final_AFFINE_REFERENCE/00_frame0001.npy')
    logs = {}
    for name in ('final_forward300', 'final_reverse300', 'final_single150', 'final16_motion300', 'final_character',
                 'final_character_full', 'final_f12_mixed', 'final_cancel_mixed',
                 'final_viewport_blended', 'final_viewport_normal', 'final_character_viewport',
                 'final16_f12_complete', 'final16_f12_cancel', 'final16_denoise_NONE',
                 'final16_denoise_SPATIAL', 'final16_denoise_TEMPORAL'):
        stderr = (root / (name + '.err')).read_text(encoding='utf-8', errors='replace')
        stdout = (root / (name + '.log')).read_text(encoding='utf-8', errors='replace')
        errors = re.findall(r'^.*(?:VUID-|Validation Error|BLI_assert failed|Traceback).*$',
                            stderr + '\n' + stdout, flags=re.M)
        # Avoid matching source/comments: only readback diagnostics carry this exact prefix.
        bypass = [float(value) for value in re.findall(
            r'DLSS RR denoise verification:.*max_error=([0-9.eE+-]+)', stderr)]
        logs[name] = {
            'rr_evaluations': len(re.findall(r'DLSS RR:.*NGX=0x00000001', stderr)),
            'max_denoise_bypass_error': max(bypass, default=None),
            'errors': errors,
        }
    result = {
        'tolerance': tolerance,
        'order_max_abs': max(entry['max_abs'] for entry in order.values()),
        'order_per_frame': order,
        'single_frame': single,
        'affine_parameter_reference': affine,
        'logs': logs,
        'visual_acceptance': 'Separate inspection required; API/numerical success is insufficient.',
    }
    failures = []
    if result['order_max_abs'] > tolerance:
        failures.append('Animation forward/reverse comparison exceeds tolerance')
    if single['max_abs'] > tolerance:
        failures.append('Single-frame rerender differs from animation beyond tolerance')
    if affine['max_abs'] > 0.002:
        failures.append('Affine parameter evaluation differs from constant reference')
    for name, log in logs.items():
        if log['errors']:
            failures.append(name + ': validation/assertion/Python errors in log')
        if log['rr_evaluations'] == 0:
            failures.append(name + ': RR was not executed')
        if log['max_denoise_bypass_error'] not in (None, 0.0):
            failures.append(name + ': hardware denoising regression')
    for name in ('final_forward300', 'final_reverse300', 'final16_motion300'):
        if logs[name]['rr_evaluations'] != 300 * 32:
            failures.append(name + ': unexpected RR evaluation count')
    result['status'] = 'FAIL' if failures else 'PASS'
    result['failures'] = failures
    (root / 'lightweight_numerical_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'order_per_frame'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    args = parser.parse_args()
    validate(args.root, args.tolerance)
