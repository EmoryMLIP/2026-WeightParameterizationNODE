#!/usr/bin/env python3
"""Write and close the manifest of a Python sweep, results/sweeps/<sweepId>.json.

The manifest records the commit, whether the code was modified, the interpreter
and library versions, and how many runs finished. collect_python.py refuses runs
whose sweep has no complete manifest. The MATLAB sweeps write theirs with
newSweep.m and finishSweep.m.

Usage: python3 sweep_manifest.py start <sweepId> <nExpected> <note>
       python3 sweep_manifest.py finish <sweepId>

finish counts the run records under results/raw/<sweepId>/python. For a sweep
whose id ends in "node-config" it also records, per dataset, the (warm start,
learning rate) with the lowest validation error.
"""
import glob
import json
import os
import re
import sys
import datetime

import torch
import torchdiffeq  # noqa: F401  (imported for its version)

from provenance import provenance

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def manifest_path(sweep):
    return os.path.join(ROOT, 'results', 'sweeps', sweep + '.json')


def run_records(sweep):
    for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'raw', sweep, 'python', '*.json'))):
        r = json.load(open(f))
        if 'test_relerr' in r and 'dataset' in r:
            yield f, r


def start(sweep, n_expected, note):
    path = manifest_path(sweep)
    if os.path.exists(path):
        m = json.load(open(path))
        m['status'] = 'running'
    else:
        p = provenance()
        slug = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', sweep)
        import importlib.metadata as md
        m = dict(sweepId=sweep, slug=slug, note=note, commit=p['commit'], dirty=p['dirty'],
                 started=p['when'], framework='python', pythonVersion=p['python'],
                 torchVersion=torch.__version__.split('+')[0],
                 torchdiffeqVersion=md.version('torchdiffeq'), host=p['host'],
                 status='running', nExpected=n_expected, nOK=0, nFail=0, finished='')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(m, open(path, 'w'), indent=2)
    print(f"sweep {sweep} started (commit {m['commit']}{' DIRTY' if m['dirty'] else ''})")


def finish(sweep):
    path = manifest_path(sweep)
    m = json.load(open(path))
    recs = list(run_records(sweep))
    m['nOK'] = len(recs)
    m['nFail'] = max(m['nExpected'] - len(recs), 0)
    m['status'] = 'complete' if m['nFail'] == 0 else 'incomplete'
    m['finished'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if sweep.endswith('node-config'):
        best = {}
        for f, r in recs:
            tag = re.search(r'_ws(\d)_lr([0-9.e-]+)_s', os.path.basename(f))
            if not tag:
                continue
            cur = best.get(r['dataset'])
            if cur is None or r['val_relerr'] < cur['val']:
                best[r['dataset']] = dict(warm_start=int(tag.group(1)), lr=tag.group(2),
                                          val=round(r['val_relerr'], 4))
        m['selected'] = best
    json.dump(m, open(path, 'w'), indent=2)
    print(f"sweep {sweep}: {m['nOK']} of {m['nExpected']} runs, status {m['status']}")


if __name__ == '__main__':
    if len(sys.argv) >= 5 and sys.argv[1] == 'start':
        start(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) == 3 and sys.argv[1] == 'finish':
        finish(sys.argv[2])
    else:
        sys.exit(__doc__)
