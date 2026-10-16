#!/usr/bin/env python3
r"""Emit figures/paired_panels.tex: the ten runs behind the basis comparison.

Reads results/runs.csv and the paper sweep manifest. Both configurations of one
comparison are drawn on a log test-error axis, with a line joining the two runs
that share a seed, so the slope of a line is that seed's paired ratio.

Panel (b) shows what a summary marker cannot carry: a ratio near 1 can come from
a consistent small effect or from seeds split evenly in both directions. The
panels also show why comparisons are paired. When the two arms correlate within
a seed, differencing removes most of the seed-to-seed spread; when they do not,
pairing gains nothing and costs degrees of freedom.

Usage: python3 export_paired_panels.py
"""
import csv
import json
import math
import os
import statistics as st
import sys

from paired_stats import paired, seed_gsd
from sweep_defaults import PAPER_SWEEP, degree_axis_sweeps, is_default_run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

# Panel (b) is fixed: the most precisely measured basis comparison. Panels (a)
# and (c) are chosen from the ledger by pick_effect() and pick_null(), so they
# follow the runs if a sweep is rerun.
SMALL = ('Legendre', dict(dataset='ELM', dynamic='hamiltonian', opti='GNvpro',
                          T='1', basis='Legendre', d='3'),
         'monomial', dict(dataset='ELM', dynamic='hamiltonian', opti='GNvpro',
                          T='1', basis='monomial', d='3'))
COMBOS = [(ds, dyn, opti) for ds in ('ELM', 'CDR', 'DCR')
          for dyn in ('ResNN', 'hamiltonian') for opti in ('sgd', 'GNvpro')]


def sweep_meta(sweep):
    m = json.load(open(os.path.join(ROOT, 'results', 'sweeps', sweep + '.json')))
    if m.get('status') != 'complete':
        sys.exit(f"sweep {sweep} is {m.get('status')!r}; its results are partial")
    if m.get('dirty'):
        sys.exit(f"sweep {sweep} was built from a dirty working tree")
    return m


def cell(rows, spec):
    out = {}
    for r in rows:
        if r['sweep'] not in degree_axis_sweeps(r['opti']) or not is_default_run(r):
            continue
        if r['precision'] != 'double' or float(r['alpha1']) != 1e-10:
            continue
        if any(r[k] != v for k, v in spec.items()):
            continue
        assert r['seed'] not in out, f"seed {r['seed']} twice in {spec}"
        out[r['seed']] = float(r['test_agg'])
    return out


def within_seed_corr(a, b):
    seeds = sorted(set(a) & set(b))
    return st.correlation([math.log(a[s]) for s in seeds],
                          [math.log(b[s]) for s in seeds])


def pick_effect(rows):
    """The largest scaling effect at T=10 whose seeds all move the same way."""
    best = None
    for ds, dyn, opti in COMBOS:
        base = dict(dataset=ds, dynamic=dyn, opti=opti, T='10', d='3')
        a = cell(rows, dict(base, basis='monomial'))
        b = cell(rows, dict(base, basis='monomial-raw'))
        seeds = sorted(set(a) & set(b))
        if len(seeds) < 3 or not (all(a[s] < b[s] for s in seeds)
                                  or all(a[s] > b[s] for s in seeds)):
            continue
        r = paired(a, b)
        if best is None or abs(math.log(r['ratio'])) > abs(math.log(best[4]['ratio'])):
            best = ('$[-1,1]$', dict(base, basis='monomial'),
                    '$[0,T]$', dict(base, basis='monomial-raw'), r)
    if best is None:
        sys.exit('no unanimous scaling comparison at T=10')
    return best[:4]


def pick_null(rows):
    """The degree-3 basis comparison closest to parity with its seeds most evenly split."""
    best, score = None, None
    for ds, dyn, opti in COMBOS:
        for T in ('1', '5', '10'):
            base = dict(dataset=ds, dynamic=dyn, opti=opti, T=T, d='3')
            a = cell(rows, dict(base, basis='Legendre'))
            b = cell(rows, dict(base, basis='monomial'))
            seeds = sorted(set(a) & set(b))
            if len(seeds) < 3:
                continue
            lower = sum(1 for s in seeds if a[s] < b[s])
            r = paired(a, b)
            sc = (abs(lower - len(seeds) / 2), abs(math.log(r['ratio'])))
            if score is None or sc < score:
                best, score = ('Legendre', dict(base, basis='Legendre'),
                               'monomial', dict(base, basis='monomial')), sc
    return best


def panel_stats(rows, spec):
    la, sa, lb, sb = spec
    a, b = cell(rows, sa), cell(rows, sb)
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 3:
        sys.exit(f"only {len(seeds)} shared seeds for {sa}")
    r = paired(a, b)
    gsd = seed_gsd(b)
    ref = math.exp(st.mean([math.log(b[s]) for s in seeds]))
    lower = sum(1 for s in seeds if a[s] < b[s])
    where = '%s, %s, %s, $T{=}%s$' % (sa['dataset'],
                                      'ResNet' if sa['dynamic'] == 'ResNN' else 'Hamiltonian',
                                      'Adam' if sa['opti'] == 'sgd' else 'GNvpro', sa['T'])
    return dict(la=la, lb=lb, a=a, b=b, seeds=seeds, r=r, gsd=gsd, ref=ref,
                lower=lower, where=where)


def main():
    commit = sweep_meta(PAPER_SWEEP)['commit']
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results', 'runs.csv'))))

    panels = [panel_stats(rows, pick_effect(rows)),
              panel_stats(rows, SMALL),
              panel_stats(rows, pick_null(rows))]
    titles = ['(a) larger than the seed spread',
              '(b) smaller, but consistent',
              '(c) no difference']

    out = os.path.join(ROOT, 'figures', 'paired_panels.tex')
    with open(out, 'w') as fh:
        fh.write('%% generated by python/export_paired_panels.py\n')
        fh.write(f'%% sweeps: {PAPER_SWEEP}, GNvpro top-up; commit {commit}\n')
        for t, p in zip(titles, panels):
            fh.write('%%   %s %s over %s, %s: ratio %.3f [%.3f, %.3f], first lower on %d of %d, '
                     'seed spread %.3f\n'
                     % (t[:3], p['la'], p['lb'], p['where'], p['r']['ratio'], p['r']['lo'],
                        p['r']['hi'], p['lower'], len(p['seeds']), p['gsd']))
        fh.write(r"""\documentclass[tikz]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{bandblue}{rgb}{0.10,0.32,0.72}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=3 by 1,horizontal sep=1.5cm},
  width=2.0in,height=2.1in,ymode=log,
  log ticks with fixed point,
  xmin=0.62,xmax=2.75,xtick={1,2},
  title style={font=\footnotesize,align=center},
  tick label style={font=\footnotesize},label style={font=\small},
  ytick pos=left,xtick pos=bottom,clip=false]
""")
        for k, (t, p) in enumerate(zip(titles, panels)):
            vals = [p['a'][s] for s in p['seeds']] + [p['b'][s] for s in p['seeds']]
            lo, hi = min(vals) / 1.08, max(vals) * 1.08
            opts = ('title={%s\\\\{\\scriptsize %s}},xticklabels={{%s},{%s}},ymin=%.6g,ymax=%.6g'
                    % (t, p['where'].replace('$T{=}', '$T='), p['la'], p['lb'], lo, hi))
            if k == 0:
                opts += ',ylabel={test relative error}'
            fh.write('\\nextgroupplot[%s]\n' % opts)
            for s in p['seeds']:
                fh.write('\\addplot[color=black,opacity=0.5,line width=0.4pt,mark=*,'
                         'mark size=1.1pt] coordinates {(1,%.6g) (2,%.6g)};\n'
                         % (p['a'][s], p['b'][s]))
            # One configuration's seed-to-seed spread, on the same axis, so the
            # size of the effect is read against the size of the noise.
            fh.write('\\draw[bandblue,|-|,line width=0.5pt] '
                     '(axis cs:2.45,%.6g) -- (axis cs:2.45,%.6g);\n'
                     % (p['ref'] / p['gsd'], p['ref'] * p['gsd']))
        fh.write('\\end{groupplot}\n\\end{tikzpicture}\n\\end{document}\n')

    print(f"wrote {out}")
    for t, p in zip(titles, panels):
        print(f"  {t[:3]} {p['la']} over {p['lb']}, {p['where']}: "
              f"ratio {p['r']['ratio']:.3f} [{p['r']['lo']:.3f}, {p['r']['hi']:.3f}], "
              f"first lower on {p['lower']} of {len(p['seeds'])}, seed spread {p['gsd']:.3f}")


if __name__ == '__main__':
    main()
