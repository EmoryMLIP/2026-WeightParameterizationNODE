"""
aggregate_fig4_traces.py

Combines the 5 per-seed nfe_trace records for each (dataset, basis) into a
single averaged trace for fig:neuralODE_loss.

Each trace is first trimmed to its own best_iter, so the iterations after the
best checkpoint are excluded from every seed before averaging.

Usage:
    python3 aggregate_fig4_traces.py --sweep 2026-10-05-node-final
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load_trimmed_trace(path):
    with open(path) as f:
        d = json.load(f)
    best_iter = d['best_iter']
    trace = [t for t in d['nfe_trace'] if t['iter'] <= best_iter]
    return trace


ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


def run_path(sweep, dataset, basis, seed):
    """Locate one run record under the sweep layout of results/README.md."""
    deg = 0 if basis == 'none' else 3
    tag = f'{dataset}_NODE_{basis}_d{deg}_T1_adam_s{seed}'
    return os.path.join(ROOT, 'results', 'raw', sweep, 'python', f'{tag}.json')


def aggregate(dataset, basis, sweep, n_seeds=5, min_seeds=2):
    by_iter = defaultdict(lambda: {'nfe': [], 'train': [], 'val': []})
    seeds_used = []
    for seed in range(n_seeds):
        path = run_path(sweep, dataset, basis, seed)
        if not os.path.exists(path):
            # Averaging over the seeds that happen to be present would give a
            # curve whose caption says five seeds and whose data is fewer.
            raise SystemExit(f'missing run: {path}\n'
                             f'The sweep is incomplete; finish it before building figures.')
        seeds_used.append(seed)
        trace = load_trimmed_trace(path)
        for t in trace:
            by_iter[t['iter']]['nfe'].append(t['nfe'])
            by_iter[t['iter']]['train'].append(t['train_relerr'])
            by_iter[t['iter']]['val'].append(t['val_relerr'])

    iters = sorted(by_iter.keys())
    result = []
    n_dropped = 0
    for it in iters:
        n = len(by_iter[it]['nfe'])
        if n < min_seeds:
            n_dropped += 1
            continue
        result.append({
            'iter': it,
            'n_seeds': n,
            'nfe_mean': float(np.mean(by_iter[it]['nfe'])),
            'train_mean': float(np.mean(by_iter[it]['train'])),
            'val_mean': float(np.mean(by_iter[it]['val'])),
            'train_std': float(np.std(by_iter[it]['train'])),
            'val_std': float(np.std(by_iter[it]['val'])),
        })

    return result, seeds_used, n_dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', required=True,
                        help='sweep id, e.g. 2026-10-05-node-final')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--min_seeds', type=int, default=2,
                         help='Drop checkpoints with fewer than this many contributing seeds '
                              '(default 2 -- the minimum needed to compute any std at all).')
    args = parser.parse_args()

    datasets = ['ELM', 'CDR', 'DCR']
    # All four arms of tables/table3.tex: the static baseline, both scaled
    # bases, and the unscaled control.
    bases = ['none', 'legendre', 'monomial', 'monomial-raw']

    out = {}
    for ds in datasets:
        for basis in bases:
            trace, seeds_used, n_dropped = aggregate(ds, basis, args.sweep, args.n_seeds, args.min_seeds)
            key = f'{ds}_{basis}'
            out[key] = trace
            min_n = min((pt['n_seeds'] for pt in trace), default=0)
            max_n = max((pt['n_seeds'] for pt in trace), default=0)
            print(f'{key}: {len(seeds_used)} seed files found, {len(trace)} checkpoints kept '
                  f'({n_dropped} dropped for having <{args.min_seeds} contributing seeds), '
                  f'seed-count per kept checkpoint ranges {min_n}-{max_n}')

    # The curve data is committed, unlike the raw run records, so the figure
    # can be rebuilt without retraining.
    outdir = os.path.join(ROOT, 'results', 'traces', args.sweep)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'node_traces.json')
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {outpath}')


if __name__ == '__main__':
    main()
