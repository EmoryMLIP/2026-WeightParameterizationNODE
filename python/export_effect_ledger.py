#!/usr/bin/env python3
r"""Emit figures/effect_ledger.tex: every choice the paper tests, on one axis.

Reads results/runs.csv and the sweep manifests in results/sweeps/. Each row is
one decision; each marker is one configuration (dataset x architecture x
optimizer x terminal time), plotted at the geometric mean over its seeds of the
paired ratio of test error. Each optimizer has its own marker shape, its own
half of the row and its own median bar, because the two disagree in sign on the
degree axis and a pooled median would report that as no effect.

The blue interval per row and optimizer is the median seed geometric SD of that
group. A marker inside it is drawn pale. Judging each marker against its own
configuration's spread instead is more exact; the figure uses the group median,
and the two differ by at most two configurations in any row (the "own" counts in
the printed summary).

Usage: python3 export_effect_ledger.py
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict
from statistics import median

from paired_stats import paired, seed_gsd
from sweep_defaults import PAPER_SWEEP, degree_axis_sweeps, is_default_run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

# (row label, numerator selector, denominator selector, split). A selector is a
# dict of ledger columns. A value of None means the column joins the cell key
# instead of being pinned, so the row spans it and contributes one dot per
# value; a set means the column joins the key but only over the listed values.
#
# `split` names a column that gets its own row rather than being pooled. It is
# used where the sign of the effect depends on that column, which a single row
# reports as a median between two opposite answers. The optimizer never needs it
# because it is drawn as the marker.
#
# The first three fields are the row's label (numerator, denominator, what the
# row is restricted to), set as a table left of the axis so both arms show.
DATASETS = ('ELM', 'CDR', 'DCR')
ROWS = [
    (r'monomial on $[0,T]$', r'monomial on $[-1,1]$', r'$T{=}10$',
     dict(basis='monomial-raw', d='3', T='10'), dict(basis='monomial', d='3', T='10'), None),
    (r'monomial on $[0,T]$', r'monomial on $[-1,1]$', r'$T{=}5$',
     dict(basis='monomial-raw', d='3', T='5'), dict(basis='monomial', d='3', T='5'), None),
    (r'monomial on $[0,T]$', r'monomial on $[-1,1]$', r'$T{=}1$',
     dict(basis='monomial-raw', d='3', T='1'), dict(basis='monomial', d='3', T='1'), None),
    (r'Legendre, $D{=}11$', r'Legendre, $D{=}0$', r'all',
     dict(basis='Legendre', d='11', T='1'), dict(basis='Legendre', d='0', T='1'), None),
    (r'Legendre, $D{=}0$', r'unconstrained', None,
     dict(basis='Legendre', d='0', T='1'), dict(basis='none', T='1'), 'dataset'),
    (r'Legendre, $D{=}4$--$8$', r'monomial, $D{=}4$--$8$', r'CDR',
     dict(basis='Legendre', d={'4', '5', '6', '7', '8'}, T='1', dataset='CDR'),
     dict(basis='monomial', d={'4', '5', '6', '7', '8'}, T='1', dataset='CDR'), None),
    (r'Legendre, $D{=}3$', r'monomial, $D{=}3$', r'all',
     dict(basis='Legendre', d='3'), dict(basis='monomial', d='3'), None),
    (r'single precision', r'double precision', r'CDR',
     dict(basis='Legendre', precision='single', T='1', d=None),
     dict(basis='Legendre', precision='double', T='1', dataset='CDR', d=None), None),
]
# The label table's columns, as fractions of the axis width to the left of it.
#
# Three units. A row is a family of comparisons reported together. A marker is
# one comparison between two configurations that differ in the one setting the
# row is about (a combination of dataset, architecture, optimizer and terminal
# time). A pair is one seed's two runs, many of them behind every marker. The
# count column carries markers and pairs.
#
# `cell` is not used for a marker: paired_stats and paper_statistics use it for
# one arm of a comparison, which is half of a marker.
COLS = (('ratio of', -1.19), ('to', -0.81), ('over', -0.42))
COUNT_X = -0.015

# Shape carries the optimizer; ink carries whether the effect clears the seed
# spread drawn for its group. Open circle against filled square keeps the two
# optimizers distinguishable in print.
MARKS = (('sgd', 'Adam', 'mark=o,mark size=2.8pt,mark options={line width=0.8pt}'),
         ('GNvpro', 'GNvpro', 'mark=square*,mark size=2.5pt'))
OFFSET = {'sgd': 0.21, 'GNvpro': -0.21}
INK = {True: 'opacity=0.9', False: 'opacity=0.28'}
OPT_LABEL = {o: lab for o, lab, _ in MARKS}


def sweep_meta(sweep):
    m = json.load(open(os.path.join(ROOT, 'results', 'sweeps', sweep + '.json')))
    if m.get('status') != 'complete':
        sys.exit(f"sweep {sweep} is {m.get('status')!r}; its results are partial")
    if m.get('dirty'):
        sys.exit(f"sweep {sweep} was built from a dirty working tree")
    return m


def select(rows, spec):
    """Rows matching spec, keyed by the columns spec leaves free."""
    out = defaultdict(dict)
    pins = {k: v for k, v in spec.items() if isinstance(v, str)}
    free = [k for k, v in spec.items() if not isinstance(v, str)]
    for r in rows:
        if r['sweep'] not in degree_axis_sweeps(r['opti']) or not is_default_run(r):
            continue
        if float(r['alpha1']) != 1e-10:
            continue
        if 'precision' not in pins and r['precision'] != 'double':
            continue
        if any(r[k] != v for k, v in pins.items()):
            continue
        if any(spec[k] is not None and r[k] not in spec[k] for k in free):
            continue
        key = (r['dataset'], r['dynamic'], r['opti'], r['T']) + tuple(r[k] for k in free)
        out[key][r['seed']] = float(r['test_agg'])
    return out


def row_points(rows, num, den):
    """One (ratio, seed gsd, optimizer, seeds) per combination both arms cover.

    One returned tuple is one marker: a comparison between the two arms at a
    combination of dataset, architecture, optimizer and terminal time. The key
    is (dataset, dynamic, opti, T, *free), so the optimizer is its third entry,
    and `seeds` is the number of matched seed pairs behind that marker, which is
    what makes a four-marker row a hundred-pair comparison.
    """
    a, b = select(rows, num), select(rows, den)
    pts = []
    for k in sorted(a.keys() & b.keys(), key=str):
        shared = set(a[k]) & set(b[k])
        if len(shared) < 3:
            continue
        pts.append((paired(a[k], b[k])['ratio'], seed_gsd(b[k]), k[2], len(shared)))
    return pts


def expand(ledger):
    """One drawn row per ROWS entry, or one per value of its split column.

    Yields (numerator, denominator, restriction, points). A repeated arm is
    blanked so the table reads as one comparison made over three datasets.
    """
    for a, b, over, num, den, split in ROWS:
        if split is None:
            yield a, b, over, row_points(ledger, num, den)
            continue
        for i, v in enumerate(DATASETS):
            pts = row_points(ledger, dict(num, **{split: v}), dict(den, **{split: v}))
            yield (a, b, v, pts) if i == 0 else ('', '', v, pts)


def main():
    commit = sweep_meta(PAPER_SWEEP)['commit']
    ledger = list(csv.DictReader(open(os.path.join(ROOT, 'results', 'runs.csv'))))

    drawn, gsds = [], []
    for a, b, over, pts in expand(ledger):
        if not pts:
            sys.exit(f"no matched combinations for row {a!r} / {b!r} / {over!r}")
        drawn.append((a, b, over, pts))
        gsds += [p[1] for p in pts]
    band = median(gsds)

    n = len(drawn)
    # Coordinate lists keyed by optimizer and by whether the effect clears the
    # group's seed spread. The optimizer also sets a vertical offset inside the
    # row, so the two groups separate without doubling the rows.
    dots = {(o, beyond): [] for o, _, _ in MARKS for beyond in (False, True)}
    meds, bands, summary = [], [], []
    for i, (_, _, _, pts) in enumerate(drawn):
        y = n - i
        for opti, _, _ in MARKS:
            sub = [p for p in pts if p[2] == opti]
            if not sub:
                continue
            yo = y + OFFSET[opti]
            g = median([p[1] for p in sub])
            for x, _, _, _ in sorted(sub):
                dots[(opti, abs(math.log(x)) > math.log(g))].append((x, yo))
            m = median([p[0] for p in sub])
            meds.append(f'({m:.4f},{yo:.2f})')
            bands.append((1 / g, g, yo))
            summary.append((i, opti, m, len(sub), g,
                            sum(1 for p in sub if abs(math.log(p[0])) > math.log(g)),
                            sum(1 for p in sub if abs(math.log(p[0])) > math.log(p[1]))))
    ticks = ','.join(str(n - i) for i in range(n))

    out = os.path.join(ROOT, 'figures', 'effect_ledger.tex')
    with open(out, 'w') as fh:
        fh.write('%% generated by python/export_effect_ledger.py\n')
        fh.write(f'%% sweeps: {PAPER_SWEEP}, GNvpro top-up on the T=1 degree axis; commit {commit}\n')
        fh.write('%% blue interval: median seed geometric SD of that row and optimizer;\n')
        fh.write('%% pale markers fall inside it. Overall median %.3f over %d combinations.\n'
                 % (band, len(gsds)))
        fh.write('%% "own" counts each marker against its own seed spread instead.\n')
        for i, opti, m, k, g, beyond, own in summary:
            lab = '%s / %s [%s] %s' % (drawn[i][0] or '"', drawn[i][1] or '"',
                                       drawn[i][2], OPT_LABEL[opti])
            fh.write(f'%%   {lab:62s} n={k:2d} median {m:.3f} gsd {g:.3f} '
                     f'beyond {beyond}/{k} (own {own}/{k})\n')
        fh.write(r"""\documentclass[tikz]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{mycolor5}{rgb}{1.0,0.03,0.0}
\definecolor{bandblue}{rgb}{0.10,0.32,0.72}
\begin{document}
\begin{tikzpicture}
\begin{axis}[width=3.6in,height=4.4in,xmode=log,
  xlabel={test error ratio, paired by seed; one marker per combination},
  xmin=0.45,xmax=32,ymin=0.4,ymax=%.2f,
  xtick={0.5,1,2,5,10,20},
  xticklabels={$0.5$,$1$,$2$,$5$,$10$,$20$},
  minor xtick={0.6,0.7,0.8,0.9,3,4,6,7,8,9,30},
  ytick={%s},yticklabels={},
  legend style={at={(0.985,0.03)},anchor=south east,draw=none,fill=none,
    font=\footnotesize,legend cell align=left},
  xmajorgrids,axis y line*=left,axis x line*=bottom,
  clip=false]
\addplot[dashed,gray,mark=none,forget plot] coordinates {(1,0.4) (1,%.2f)};
""" % (n + 1.25, ticks, n + 1.25))
        # The seed-spread interval, one per row and optimizer, drawn before the
        # markers so they sit on top of it.
        for lo, hi, y in bands:
            fh.write('\\draw[bandblue,|-|,line width=0.4pt] '
                     '(axis cs:%.4f,%.2f) -- (axis cs:%.4f,%.2f);\n' % (lo, y, hi, y))
        # \addlegendimage draws a legend key with no data behind it, so the
        # axis can be unclipped for the label table without a dummy plot
        # appearing outside the frame.
        for _, lab, style in MARKS:
            fh.write('\\addlegendimage{only marks,%s,color=black,%s}\n'
                     '\\addlegendentry{%s}\n' % (style, INK[True], lab))
        for (opti, beyond), coords in dots.items():
            if not coords:
                continue
            style = next(s for o, _, s in MARKS if o == opti)
            fh.write('\\addplot[only marks,%s,color=black,%s,forget plot] '
                     'coordinates {%s};\n'
                     % (style, INK[beyond],
                        ' '.join(f'({x:.4f},{y:.2f})' for x, y in coords)))
        fh.write('\\addplot[only marks,mark=|,mark size=5.5pt,line width=1.5pt,'
                 'color=mycolor5,forget plot] coordinates {%s};\n' % ' '.join(meds))

        # The row labels are set as a table to the left of the axis rather than
        # as y tick labels, so that both arms of every ratio and the number of
        # paired configurations line up in columns. Positions are fractions of
        # the axis width and height, which is why the axis has a fixed size.
        def yfrac(y):
            return (y - 0.4) / ((n + 1.25) - 0.4)

        head = n + 0.50
        for title, x in COLS:
            fh.write('\\node[anchor=base west,font=\\footnotesize\\itshape] '
                     'at (axis description cs:%.3f,%.4f) {%s};\n'
                     % (x, yfrac(head), title))
        for line, text in enumerate(('combinations', '(seed pairs)')):
            fh.write('\\node[anchor=base east,font=\\footnotesize\\itshape] '
                     'at (axis description cs:%.3f,%.4f) {%s};\n'
                     % (COUNT_X, yfrac(head + 0.42 - 0.42 * line), text))
        for i, (a, b, over, pts) in enumerate(drawn):
            y = yfrac(n - i - 0.14)
            for text, (_, x) in zip((a, b, over), COLS):
                if not text:
                    continue
                fh.write('\\node[anchor=base west,font=\\footnotesize] '
                         'at (axis description cs:%.3f,%.4f) {%s};\n' % (x, y, text))
            fh.write('\\node[anchor=base east,font=\\footnotesize] '
                     'at (axis description cs:%.3f,%.4f) {%d (%d)};\n'
                     % (COUNT_X, y, len(pts), sum(p[3] for p in pts)))
        fh.write('\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n')

    print(f"wrote {out}")
    print(f"  median seed geometric SD over all {len(gsds)} combinations: {band:.3f}")
    for i, opti, m, k, g, beyond, own in summary:
        a, b, over = drawn[i][0], drawn[i][1], drawn[i][2]
        lab = '%-22s / %-22s [%s]' % (a or '"', b or '"', over)
        print(f"  {lab:54s} {OPT_LABEL[opti]:7s} n={k:2d} median {m:.3f} "
              f"gsd {g:.3f} beyond {beyond}/{k} (own {own}/{k})")


if __name__ == '__main__':
    main()
