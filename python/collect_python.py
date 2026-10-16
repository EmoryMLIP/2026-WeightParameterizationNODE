#!/usr/bin/env python3
"""Walk results/raw/<sweep>/python/*.json into the ledger results/runs_python.csv.

One row per run, across every sweep. This is the optimize-then-discretize
counterpart of matlab/analysis/collectResults.m, which writes runs.csv. The two
ledgers are separate because the halves measure different things (NFE exists
only here; nt and GNvpro only there), but they share sweep ids, the run-tag
convention and the error metric, so rows compare by (dataset, basis, d, seed).

A sweep whose manifest is missing, not complete, or built from a dirty tree is
refused, as is ragged seed coverage.

Usage:  python3 collect_python.py [--raw-root ../results/raw]
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import dataset_config

COLUMNS = [
    'sweep', 'commit', 'dataset', 'dynamic', 'basis', 'd', 'T', 'opti', 'seed',
    'gradient', 'loss', 'alpha', 'nTrain', 'nVal', 'nTest', 'nc', 'hidden',
    'rtol', 'atol', 'niters', 'bestIter', 'stoppedEarly',
    'train_agg', 'val_agg', 'test_agg',
    'train_nfe', 'val_nfe', 'test_nfe',
    'trivial_train_agg', 'trivial_val_agg', 'trivial_test_agg',
    'nParamsTotal',
]


def row_from(run, sweep, commit):
    triv = run.get('trivial_relerr') or {}
    # Split sizes are fixed by the dataset and SPLIT_SEED, so they come from
    # dataset_config.py rather than from each run record.
    cfg = dataset_config.get(run['dataset'])
    return {
        'sweep': sweep, 'commit': commit,
        'dataset': run['dataset'], 'dynamic': 'NODE', 'basis': run['basis'],
        'd': run['degree'], 'T': run['T'], 'opti': 'adam', 'seed': run['seed'],
        'gradient': run.get('gradient', 'unspecified'),
        'loss': run.get('loss', 'unspecified'),
        'alpha': run.get('alpha', ''),
        'nTrain': run.get('n_train', cfg['nTrain'] if cfg['nTrain'] else 'file'),
        'nVal': run.get('n_val', cfg['nVal'] if cfg['nVal'] else 'file'),
        'nTest': run.get('n_test', ''),
        'nc': run.get('nc', cfg['nc']), 'hidden': run.get('hidden', ''),
        'rtol': run.get('rtol', ''), 'atol': run.get('atol', ''),
        'niters': run.get('niters_requested', ''),
        'bestIter': run.get('best_iter', ''),
        'stoppedEarly': run.get('stopped_early', ''),
        'train_agg': run['train_relerr'], 'val_agg': run['val_relerr'],
        'test_agg': run['test_relerr'],
        'train_nfe': run.get('train_nfe', ''), 'val_nfe': run.get('val_nfe', ''),
        'test_nfe': run.get('test_nfe', ''),
        'trivial_train_agg': triv.get('train', ''),
        'trivial_val_agg': triv.get('val', ''),
        'trivial_test_agg': triv.get('test', ''),
        'nParamsTotal': run.get('trainable_params', ''),
    }


def check_provenance(rows, sweeps_dir):
    """Refuse anything that cannot be attributed to a committed code state."""
    problems = []
    by_sweep = defaultdict(list)
    for r in rows:
        by_sweep[r['sweep']].append(r)

    for sweep, rs in sorted(by_sweep.items()):
        mpath = os.path.join(sweeps_dir, sweep + '.json')
        if not os.path.exists(mpath):
            problems.append(f"{sweep}: no manifest at {mpath}")
            continue
        m = json.load(open(mpath))
        if m.get('status') != 'complete':
            problems.append(f"{sweep}: manifest status is {m.get('status')!r}, "
                            "so its results are partial and must not feed a table")
        if m.get('dirty') and 'provenanceNote' not in m:
            problems.append(f"{sweep}: built from a dirty working tree, with no provenanceNote "
                            "recording what was uncommitted and where that code landed")

        commits = {r['commit'] for r in rs}
        if len(commits) > 1 and 'provenanceNote' not in m:
            problems.append(f"{sweep}: spans commits {sorted(commits)} with no "
                            "provenanceNote in the manifest explaining why")

        # Ragged seed coverage would silently change the number of seeds behind
        # a mean. Every (dataset, basis, d) cell must carry the same seeds.
        cells = defaultdict(set)
        for r in rs:
            cells[(r['dataset'], r['basis'], r['d'])].add(r['seed'])
        sizes = {frozenset(v) for v in cells.values()}
        if len(sizes) > 1:
            counts = sorted({len(v) for v in cells.values()})
            problems.append(f"{sweep}: ragged seed coverage across cells "
                            f"(cell sizes {counts})")
    return problems


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', default=os.path.join(root, 'results', 'raw'))
    ap.add_argument('--out', default=os.path.join(root, 'results', 'runs_python.csv'))
    ap.add_argument('--sweeps-dir', default=os.path.join(root, 'results', 'sweeps'))
    args = ap.parse_args()

    rows = []
    for sweep_dir in sorted(glob.glob(os.path.join(args.raw_root, '*', 'python'))):
        sweep = os.path.basename(os.path.dirname(sweep_dir))
        if sweep == 'scratch':
            continue
        mpath = os.path.join(args.sweeps_dir, sweep + '.json')
        commit = json.load(open(mpath)).get('commit', 'unknown') if os.path.exists(mpath) else 'unknown'
        skipped = 0
        for f in sorted(glob.glob(os.path.join(sweep_dir, '*.json'))):
            run = json.load(open(f))
            # A run record carries both a dataset and a test error. Files
            # without them are aggregates (averaged traces).
            if 'test_relerr' not in run or 'dataset' not in run:
                skipped += 1
                continue
            rows.append(row_from(run, sweep, commit))
        if skipped:
            print(f"  {sweep}: skipped {skipped} file(s) that are not run records")

    if not rows:
        sys.exit("no python runs found under " + args.raw_root)

    problems = check_provenance(rows, args.sweeps_dir)
    if problems:
        print("PROVENANCE PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: (r['sweep'], r['dataset'], r['basis'], r['d'], r['seed']))
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs to {args.out}")


if __name__ == '__main__':
    main()
