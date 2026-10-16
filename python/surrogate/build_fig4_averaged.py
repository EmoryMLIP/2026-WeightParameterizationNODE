"""
build_fig4_averaged.py -- builds figures/node_nfe_{elm,cdr,dcr}.pdf
(fig:neuralODE_loss) from the 5-seed averaged traces
results/traces/<sweep>/node_traces.json, produced by aggregate_fig4_traces.py.

Usage:
    python3 aggregate_fig4_traces.py --sweep <id>
    python3 build_fig4_averaged.py --sweep <id>
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.unicode_minus': False,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
_ap = argparse.ArgumentParser()
_ap.add_argument('--sweep', required=True, help='sweep id, e.g. 2026-10-05-node-final')
_args = _ap.parse_args()
_tracefile = os.path.join(ROOT, 'results', 'traces', _args.sweep, 'node_traces.json')
if not os.path.exists(_tracefile):
    raise SystemExit(f'{_tracefile} not found -- run aggregate_fig4_traces.py --sweep {_args.sweep} first')
with open(_tracefile) as f:
    all_traces = json.load(f)

def get(dataset, basis):
    trace = all_traces[f'{dataset}_{basis}']
    nfe = np.array([t['nfe_mean'] for t in trace])
    train = np.array([t['train_mean'] for t in trace])
    val = np.array([t['val_mean'] for t in trace])
    train_std = np.array([t['train_std'] for t in trace])
    val_std = np.array([t['val_std'] for t in trace])
    n_seeds = [t['n_seeds'] for t in trace]
    return nfe, train, val, train_std, val_std, n_seeds

colors = {'monomial': '#7B3FA0', 'legendre': '#2E8B57', 'none': '#1F6FB2'}
labels = {'monomial': 'monomial', 'legendre': 'Legendre', 'none': 'non-parameterized'}
# The dataset is in the filename, so main.tex selects each panel by name.
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')
panel_files = {ds: os.path.join(FIGDIR, f'node_nfe_{ds.lower()}.pdf')
               for ds in ('ELM', 'CDR', 'DCR')}

def blend_with_white(color, alpha):
    # Pre-blend with white instead of passing alpha, so the band renders the
    # same in backends without transparency.
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple(alpha * rgb + (1 - alpha) * white)

for dataset, outfile in panel_files.items():
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    min_seeds_seen = 5
    for basis in ['monomial', 'legendre', 'none']:
        nfe, train, val, train_std, val_std, n_seeds = get(dataset, basis)
        min_seeds_seen = min(min_seeds_seen, min(n_seeds))
        c = colors[basis]
        c_light = blend_with_white(c, 0.12)
        ax.plot(nfe, train, color=c, linestyle='-', linewidth=1.4,
                label=f'{labels[basis]} (train)')
        train_lower = np.maximum(train - train_std, train * 1e-2)
        ax.fill_between(nfe, train_lower, train + train_std, color=c_light)
        ax.plot(nfe, val, color=c, linestyle='--', linewidth=1.4,
                label=f'{labels[basis]} (val)')
        val_lower = np.maximum(val - val_std, val * 1e-2)
        ax.fill_between(nfe, val_lower, val + val_std, color=c_light)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('no. function evaluations')
    ax.set_ylabel('relative error')
    ax.set_title(dataset)
    ax.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, which='major', alpha=0.35, linewidth=0.6)
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {outfile}  (min seeds contributing to any single checkpoint: {min_seeds_seen})')
