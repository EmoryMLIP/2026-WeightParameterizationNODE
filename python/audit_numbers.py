#!/usr/bin/env python3
"""Check that every number quoted in the results sections traces to the ledger.

Extracts every numeric literal from the results and discussion sections of
main.tex and reports those that appear in neither ledger, in no table body, and
in no set of quantities that are computed rather than measured.

A flagged number is not necessarily wrong (aggregate statistics such as a median
ratio are computed from the ledger, not stored in it), but each needs a stated
reason. Exit status is 1 if any literal has no ledger entry and no statistical
context.

Usage: python3 audit_numbers.py [--section SECT]
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

# Quantities that are derived in the text or fixed by the experimental design,
# not read from a run record. Each needs a source the reader can check.
DERIVED = {
    # conditioning, from getPolynomialBasis / testBasis and the Python mirror
    '7.16', '7.2', '2.29', '2.3', '102', '350', '349', '1640', '123', '67.6', '5.90',
    # experimental design: splits, widths, budgets, optimizer settings
    '12', '11', '5', '3', '32', '0.001', '400', '200', '8000', '1000', '1740', '87',
    '15', '16', '10000', '2000', '20', '36', '72', '24', '19', '47', '23', '6000',
    '9400', '500', '100', '249', '497',
    # problem definitions: dimensions and PDE coefficients, from Section 5.1
    '55', '882', '4.5', '0.8', '5.5', '0.5', '0.25', '2', '21', '1', '0',
    # arithmetic quoted in Section 4.3
    '1.6', '13', '1e-13', '1e-3', '10',
}
# Aggregate statistics are computed from the ledger and so are not in it. Each
# must be reproducible from a named sweep (see paper_statistics.py).
STATISTICS = re.compile(r'median|mean|ratio|standard deviation|pooled|factors? of|%|ranging|range')
NUM = re.compile(r'(?<![\w.])(\d+\.\d+|\d+)(?![\w.])')
# The endpoints of a confidence interval, written [$lo$, $hi$]. They belong to
# the statistic the interval qualifies, whose name can lie outside the context
# window.
INTERVAL = re.compile(r'\[\$\d+(?:\.\d+)?\$,\s*\$\d+(?:\.\d+)?\$\]')
# The estimate an interval qualifies, as in "$0.71$ [$0.65$, $0.78$]". Same
# reason as INTERVAL.
ESTIMATE = re.compile(r'^\$?\s*\[\$')


def ledger_values():
    vals = set()
    for name in ('runs.csv', 'runs_python.csv'):
        p = os.path.join(ROOT, 'results', name)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            for k, v in r.items():
                if not v:
                    continue
                try:
                    f = float(v)
                except ValueError:
                    continue
                for d in (3, 4):
                    vals.add(f'{f:.{d}f}'.rstrip('0').rstrip('.'))
                vals.add(str(int(f)) if f == int(f) else '')
    for t in ('tableC1.tex', 'table3.tex'):
        p = os.path.join(ROOT, 'tables', t)
        if os.path.exists(p):
            vals.update(NUM.findall(open(p).read()))
    return vals - {''}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-marker', default=r'\section{Computational results}')
    # The end of the prose, not of the paper: the Tables and Figures sections
    # hold generated bodies whose numbers are checked by regenerating them.
    ap.add_argument('--to-marker', default=r'\section{Tables}')
    a = ap.parse_args()
    tex = open(os.path.join(ROOT, 'main.tex')).read()
    i, j = tex.index(a.from_marker), tex.index(a.to_marker)
    body = tex[i:j]
    # drop comments and \label/\ref, which carry digits that are not results
    body = body.replace('{,}', '')   # 10{,}000 must scan as one number
    body = re.sub(r'(?<!\\)%.*', '', body)
    body = re.sub(r'\\(label|ref|eqref|cite)\{[^}]*\}', '', body)

    known = ledger_values() | DERIVED
    intervals = [m.span() for m in INTERVAL.finditer(body)]
    stats, unexplained = [], []
    for m in NUM.finditer(body):
        v = m.group(1)
        norm = v.rstrip('0').rstrip('.') if '.' in v else v
        if norm in known or v in known:
            continue
        ctx = ' '.join(body[max(0, m.start() - 70):m.end() + 25].split())
        paired = (any(lo <= m.start() < hi for lo, hi in intervals)
                  or ESTIMATE.match(body[m.end():m.end() + 4]))
        (stats if paired or STATISTICS.search(ctx) else unexplained).append((v, ctx))

    if stats:
        print(f"{len(stats)} aggregate statistic(s) -- computed from the ledger, "
              "not stored in it:\n")
        for v, ctx in stats:
            print(f"  {v:>10}  ...{ctx}...")
    print(f"\n{len(unexplained)} literal(s) with NO ledger entry and no statistical "
          "context -- check each:\n")
    for v, ctx in unexplained:
        print(f"  {v:>10}  ...{ctx}...")
    return 1 if unexplained else 0


if __name__ == '__main__':
    sys.exit(main())
