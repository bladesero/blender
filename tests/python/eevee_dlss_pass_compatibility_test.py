# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Run with host Python and --blender pointing to a Vulkan/NGX-enabled installation.

Check actual NGX evaluation logs: successful rendering alone can mean native fallback.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).with_name('eevee_dlss_compatibility.py')
    results = []
    for mode in ('SR', 'RR'):
        for case, blocked_pass in (('LEGACY_PASSES', None), ('LEGACY_WITH_DEPTH', 'Depth'),
                                   ('LEGACY_WITH_TRANSPARENT', 'Transparent')):
            name = mode + '_' + case
            command = [str(args.blender), '--background', '--factory-startup', '--gpu-backend', 'vulkan',
                       '--debug-gpu', '--python-exit-code', '1', '--python', str(fixture), '--',
                       '--case', case, '--mode', mode, '--quality', 'BALANCED', '--width', '320',
                       '--height', '240', '--samples', '32', '--sequence', '1,2,1',
                       '--output', str(args.output / name)]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       encoding='utf-8', errors='replace', timeout=180)
            log = completed.stdout
            (args.output / (name + '.log')).write_text(log, encoding='utf-8')
            evaluations = len(re.findall(r'DLSS ' + mode + r':.*NGX=0x00000001', log))
            errors = re.findall(r'^.*(?:VUID-|Validation Error|BLI_assert failed|Traceback).*$', log, re.M)
            assert completed.returncode == 0 and not errors, (name, completed.returncode, errors)
            frames = json.loads((args.output / name / 'results.json').read_text())
            assert [frame['frame'] for frame in frames] == [1, 2, 1], name
            if blocked_pass is None:
                assert evaluations == 96 and 'actual Native' not in log, name
            else:
                assert evaluations == 0 and 'actual Native' in log, name
                diagnostics = re.findall(r'^.*\[DATA_PASSES\].*$', log, re.M)
                assert diagnostics and all(line.endswith('enabled data passes: ' + blocked_pass)
                                           for line in diagnostics), (name, diagnostics)
            results.append({'case': name, 'evaluations': evaluations, 'expected_blocker': blocked_pass,
                            'frames': len(frames), 'status': 'PASS'})
            print(name, 'PASS', flush=True)
    (args.output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
