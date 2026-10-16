#!/usr/bin/env python3
r"""Emit tables/table3.tex from the neural-ODE ledger results/runs_python.csv.

The manuscript \inputs this file, so no number is transcribed. The file starts
with a provenance header naming the sweep and the commit behind it.

Reads the ledger rather than the per-run JSONs so that a table is built only
from runs that passed collect_python.py's provenance checks.

Usage:  python3 collect_python.py && python3 make_tables.py
"""
import argparse
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict


from provenance import provenance_line

BASIS_LABEL = {
    'none': 'static ($D=0$)',
    'monomial': 'monomial',
    'legendre': 'Legendre',
    'monomial-raw': 'monomial, unscaled',
}
BASIS_ORDER = ['none', 'legendre', 'monomial', 'monomial-raw']


def mean_std(vals):
    # statistics rather than numpy, so the table builds without a scientific stack.
    v = [float(x) for x in vals]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def block_exponent(values):
    """The power of ten to factor out of one dataset's block.

    The datasets' errors differ by orders of magnitude, so one fixed number of
    decimals would print small errors and standard deviations as identical or
    zero. The exponent is factored out once per block and the rows carry
    mantissas.
    """
    return int(math.floor(math.log10(max(values))))


def write_table3(ledger, out_dir, sweep=None):
    rows = [r for r in csv.DictReader(open(ledger))]
    if sweep:
        rows = [r for r in rows if r['sweep'] == sweep]
    else:
        sweeps = sorted({r['sweep'] for r in rows})
        if len(sweeps) > 1:
            sys.exit(f"ledger spans sweeps {sweeps}; pass --sweep to pick one. "
                     "Mixing sweeps in one table combines cells from "
                     "different code versions.")
        sweep = sweeps[0]
    if not rows:
        sys.exit(f"no runs for sweep {sweep!r} in {ledger}")

    commits = sorted({r['commit'] for r in rows})
    grads = sorted({r['gradient'] for r in rows})
    losses = sorted({r['loss'] for r in rows})
    alphas = sorted({r['alpha'] for r in rows})

    cells = defaultdict(list)
    for r in rows:
        cells[(r['dataset'], r['basis'])].append(r)

    path = os.path.join(out_dir, 'table3.tex')
    with open(path, 'w') as fh:
        print(provenance_line('%'), file=fh)
        print(f'% source: results/{os.path.basename(ledger)}   sweep: {sweep}', file=fh)
        print('% regenerate with: make tables', file=fh)
        print('% metric: aggregate relative error ||Pred-C||_F/||C||_F on the test set', file=fh)
        print('% cells are mean +/- std over the seeds noted per row, as mantissas'
              ' against the power of ten named in each block', file=fh)
        print(f"% commit: {', '.join(commits)}   gradient: {', '.join(grads)}", file=fh)
        print(f"% loss: {', '.join(losses)}   alpha: {', '.join(alphas)}", file=fh)
        print('\\begin{tabular}{llccccc}', file=fh)
        print('\\toprule', file=fh)
        print('Dataset & Parameterization & Params & Train & Val & Test & Test NFE \\\\',
              file=fh)
        print('\\midrule', file=fh)

        for ds in ('ELM', 'CDR', 'DCR'):
            present = [b for b in BASIS_ORDER if (ds, b) in cells]
            if not present:
                continue
            exp = block_exponent([float(r[k]) for b in present for r in cells[(ds, b)]
                                  for k in ('train_agg', 'val_agg', 'test_agg')])
            scale = 10.0 ** exp
            head = f'{ds} $(\\times 10^{{{exp}}})$'
            for basis in present:
                g = cells[(ds, basis)]
                out = []
                for k in ('train_agg', 'val_agg', 'test_agg'):
                    m, sd = mean_std([r[k] for r in g])
                    out.append(f'${m / scale:.2f} \\pm {sd / scale:.2f}$')
                nm, ns = mean_std([r['test_nfe'] for r in g])
                params = g[0]['nParamsTotal']
                print(f"{head if basis == present[0] else ''} & {BASIS_LABEL[basis]} & {params} & "
                      + ' & '.join(out)
                      + f' & ${nm:.0f} \\pm {ns:.0f}$ \\\\  % n={len(g)} seeds', file=fh)
            triv = g[0].get('trivial_test_agg')
            if triv:
                print('& \\emph{mean predictor} & & & & $%.2f$ & \\\\'
                      % (float(triv) / scale), file=fh)
            print('\\midrule' if ds != 'DCR' else '\\bottomrule', file=fh)
        print('\\end{tabular}', file=fh)
    print(f'wrote {path}  ({len(rows)} runs, sweep {sweep})')


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger', default=os.path.join(root, 'results', 'runs_python.csv'))
    ap.add_argument('--out-dir', default=os.path.join(root, 'tables'))
    ap.add_argument('--sweep', default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    write_table3(a.ledger, a.out_dir, a.sweep)
