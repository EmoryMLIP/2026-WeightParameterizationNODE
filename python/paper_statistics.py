#!/usr/bin/env python3
"""Recompute every aggregate statistic the results sections quote.

The paper quotes derived numbers that are computed from the ledger rather than
stored in it, so audit_numbers.py can only confirm that they are statistics.
This prints them, labelled, so the prose can be checked against them.

Every comparison is paired by seed on log test error (see paired_stats.py for
why). A difference is claimed when the 95% interval of the ratio excludes 1,
and a ratio is within +-10% of parity when the whole interval lies inside that
margin; no multiple-comparison correction is applied. The exact sign-flip p is
printed as a sensitivity check.

Usage: python3 paper_statistics.py        (RUNS_CSV=<path> for another ledger copy)
"""
import csv
import math
import os
from collections import defaultdict
from statistics import mean, median

from paired_stats import differs, mde, paired, seed_gsd, signflip_p, within
from sweep_defaults import PAPER_SWEEP, degree_axis_sweeps, is_default_run

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MARGIN = 0.10
ARCH = (('ResNN', 'ResNet'), ('hamiltonian', 'Hamiltonian'))
OPT = (('sgd', 'Adam'), ('GNvpro', 'GNvpro'))
DS = ('ELM', 'CDR', 'DCR')


def load():
    path = os.environ.get('RUNS_CSV') or os.path.join(ROOT, 'results', 'runs.csv')
    return list(csv.DictReader(open(path)))


def cells(rows, sweeps=lambda r: {PAPER_SWEEP}, precision='double', alpha='1e-10', field='test_agg'):
    """(dataset, dynamic, opti, T, basis, d) -> {seed: value}."""
    out = defaultdict(dict)
    for r in rows:
        if r['sweep'] not in sweeps(r) or not is_default_run(r):
            continue
        if r['precision'] != precision or float(r['alpha1']) != float(alpha):
            continue
        key = (r['dataset'], r['dynamic'], r['opti'], r['T'], r['basis'], r['d'])
        assert r['seed'] not in out[key], f"seed {r['seed']} twice in {key}"
        out[key][r['seed']] = float(r[field])
    return out


def family(title, pairs, margin=MARGIN):
    """pairs: (label, a, b) with a, b as {seed: value}. Prints a/b per pair, flagged by the interval criterion.

    Each line also carries the smallest ratio that cell could have resolved
    (mde) and the seed-to-seed spread of its reference arm (gsd). A null is only
    as strong as its mde, and an effect only matters where it exceeds gsd.
    """
    print(f"\n{title}")
    res = [(lab, paired(a, b), signflip_p(a, b), seed_gsd(b)) for lab, a, b in pairs if a and b]
    if not res:
        print("  NO PAIRS")
        return []
    rej = [differs(r) for _, r, _, _ in res]
    for (lab, r, sf, gsd), rj in zip(res, rej):
        flag = 'DIFF' if rj else ('within' if within(r, margin) else '')
        print(f"  {lab:34} n={r['n']:2d} ratio {r['ratio']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] "
              f"t {r['t']:+6.2f} p {r['p']:.4f} signflip {sf:.4f} "
              f"mde {mde(r):.3f} gsd {gsd:.3f} {flag}")
    rat = [r['ratio'] for _, r, _, _ in res]
    md = [mde(r) for _, r, _, _ in res]
    gs = [gsd for _, _, _, gsd in res]
    lower = sum(1 for (_, r, _, _), j in zip(res, rej) if j and r['ratio'] < 1)
    higher = sum(1 for (_, r, _, _), j in zip(res, rej) if j and r['ratio'] > 1)
    print(f"  family {len(res)}: median ratio {median(rat):.3f}, range {min(rat):.2f} to {max(rat):.2f}; "
          f"interval excludes 1: {sum(rej)} "
          f"(ratio<1: {lower}, ratio>1: {higher}); interval within {100 * margin:.0f}%: "
          f"{sum(1 for _, r, _, _ in res if within(r, margin))}")
    below = sum(1 for _, r, _, _ in res if r['ratio'] < 1)
    print(f"  direction: lower in {below} of {len(res)}, higher in {len(res) - below}")
    print(f"  detectable: median mde {median(md):.3f}, range {min(md):.3f} to {max(md):.3f}; "
          f"seed gsd median {median(gs):.3f}, range {min(gs):.3f} to {max(gs):.3f}; "
          f"cells whose ratio exceeds their own seed gsd: "
          f"{sum(1 for (_, r, _, gsd) in res if abs(math.log(r['ratio'])) > math.log(gsd))}")
    return [(lab, r, j) for (lab, r, _, _), j in zip(res, rej)]


def main():
    rows = load()
    g = cells(rows)
    print("=" * 72)
    print("PAIRED COMPARISONS: log test error, matched seeds; 95% interval excludes 1 (DIFF) or lies within +-10% (within)")

    family("BASIS d=3: Legendre / monomial, all T   [Section 5.2, basis]",
           [(f"{ds} {al} {ol} T={T}", g.get((ds, dyn, op, T, 'Legendre', '3')), g.get((ds, dyn, op, T, 'monomial', '3')))
            for ds in DS for dyn, al in ARCH for op, ol in OPT for T in ('1', '5', '10')])

    family("BASIS BY DEGREE, CDR T=1: Legendre / monomial   [Section 5.2, conditioning gap]",
           [(f"CDR d={d} {al} {ol}", g.get(('CDR', dyn, op, '1', 'Legendre', str(d))), g.get(('CDR', dyn, op, '1', 'monomial', str(d))))
            for d in range(3, 9) for dyn, al in ARCH for op, ol in OPT])

    res = family("SCALING d=3: unscaled monomial / scaled monomial   [Section 5.2, scaling]",
                 [(f"T={T} {ds} {al} {ol}", g.get((ds, dyn, op, T, 'monomial-raw', '3')), g.get((ds, dyn, op, T, 'monomial', '3')))
                  for T in ('1', '5', '10') for ds in DS for dyn, al in ARCH for op, ol in OPT])
    for T in ('1', '5', '10'):
        sub = [(r, j) for lab, r, j in res if lab.startswith(f"T={T} ")]
        rat = [r['ratio'] for r, _ in sub]
        print(f"  T={T:>2}: median ratio {median(rat):.3f}, max {max(rat):.2f}; interval excludes 1, unscaled worse in "
              f"{sum(1 for r, j in sub if j and r['ratio'] > 1)}/{len(sub)}, better in "
              f"{sum(1 for r, j in sub if j and r['ratio'] < 1)}")

    s = cells(rows, precision='single')
    family("PRECISION, CDR T=1 Legendre d=3..11: single / double   [Section 5.2, precision]",
           [(f"d={d} {al} {ol}", s.get(('CDR', dyn, op, '1', 'Legendre', str(d))), g.get(('CDR', dyn, op, '1', 'Legendre', str(d))))
            for d in range(3, 12) for dyn, al in ARCH for op, ol in OPT])

    print("\nALPHA, Legendre d=3 T=1: validation error at alpha / at 1e-10   [Section 5.2, regularization]")
    base = cells(rows, field='val_agg')
    for a in ('1e-08', '1e-06', '1e-05', '0.0001'):
        alt = cells(rows, alpha=a, field='val_agg')
        family(f"  alpha={a}",
               [(f"{ds} {al} {ol}", alt.get((ds, dyn, op, '1', 'Legendre', '3')), base.get((ds, dyn, op, '1', 'Legendre', '3')))
                for ds in DS for dyn, al in ARCH for op, ol in OPT])

    lv = cells(rows, sweeps=lambda r: degree_axis_sweeps(r['opti']))
    for op, ol in OPT:
        family(f"DEGREE ENDPOINTS, T=1 {ol}: D=0 / unconstrained   [Section 5.2, parameter saving]",
               [(f"{ds} {al}", lv.get((ds, dyn, op, '1', 'Legendre', '0')), lv.get((ds, dyn, op, '1', 'none', '0')))
                for ds in DS for dyn, al in ARCH])
        family(f"DEGREE ENDPOINTS, T=1 {ol}: D=11 / unconstrained   [Section 5.2, same function class]",
               [(f"{ds} {al}", lv.get((ds, dyn, op, '1', 'Legendre', '11')), lv.get((ds, dyn, op, '1', 'none', '0')))
                for ds in DS for dyn, al in ARCH])
        family(f"DEGREE ENDPOINTS, T=1 {ol}: D=11 / D=0   [Section 5.2, accuracy]",
               [(f"{ds} {al}", lv.get((ds, dyn, op, '1', 'Legendre', '11')), lv.get((ds, dyn, op, '1', 'Legendre', '0')))
                for ds in DS for dyn, al in ARCH])

    conv = cells(rows, sweeps=lambda r: {'2026-10-05-convergence'})
    fam = []
    for ds in ('CDR', 'DCR'):
        for dyn, al in ARCH:
            for arm, b, d in (('unc', 'none', '0'), ('D0', 'Legendre', '0'), ('D11', 'Legendre', '11')):
                fam.append((f"{ds} {al} {arm}", conv.get((ds, dyn, 'sgd', '1', b, d)), g.get((ds, dyn, 'sgd', '1', b, d))))
    family("FIVE TIMES THE BUDGET, Adam T=1: 5x budget / paper budget   [Section 5.2, limitations]", fam)
    for ds in ('CDR', 'DCR'):
        for dyn, al in ARCH:
            u, e = conv.get((ds, dyn, 'sgd', '1', 'none', '0')), conv.get((ds, dyn, 'sgd', '1', 'Legendre', '11'))
            ub, eb = g.get((ds, dyn, 'sgd', '1', 'none', '0')), g.get((ds, dyn, 'sgd', '1', 'Legendre', '11'))
            if u and e:
                r5, r1 = paired(e, u), paired({k: eb[k] for k in e}, {k: ub[k] for k in u})
                print(f"  {ds} {al}: D=11/unconstrained at budget {r1['ratio']:.3f} [{r1['lo']:.3f}, {r1['hi']:.3f}], "
                      f"at 5x {r5['ratio']:.3f} [{r5['lo']:.3f}, {r5['hi']:.3f}]  (seeds {sorted(e)})")

    print("\nBUDGET: best validation iterate as a fraction of the budget   [Section 5.2, limitations]")
    for ds in DS:
        rs = [r for r in rows if r['sweep'] == PAPER_SWEEP and is_default_run(r) and r['dataset'] == ds
              and r['precision'] == 'double' and float(r['alpha1']) == 1e-10]
        fr = [float(r['bestEpoch']) / float(r['maxEpochs']) for r in rs if r['opti'] == 'sgd']
        it = [float(r['bestEpoch']) for r in rs if r['opti'] == 'GNvpro']
        print(f"  {ds}: Adam median {100 * median(fr):.0f}%, final twentieth {100 * mean(x > 0.95 for x in fr):.0f}% "
              f"(n={len(fr)}); GNvpro median best iterate {median(it):.0f}")

    print("\nDCR test / training error over T=1 Legendre and unconstrained cells   [Section 5.2, limitations]")
    tr = cells(rows, sweeps=lambda r: degree_axis_sweeps(r['opti']), field='train_agg')
    rr = [mean(v.values()) / mean(tr[k].values()) for k, v in lv.items()
          if k[0] == 'DCR' and k[3] == '1' and k[4] in ('Legendre', 'none')]
    print(f"  range {min(rr):.2f} to {max(rr):.2f} over {len(rr)} cells")

    node_report()

    print("\nWALL CLOCK, Legendre d=3 / unconstrained, same T   [Section 5.2]")
    tm = cells(rows, field='elapsedSeconds')
    keys = [((ds, dyn, op, T, 'Legendre', '3'), (ds, dyn, op, T, 'none', '0'))
            for ds in DS for dyn, _ in ARCH for op, _ in OPT for T in ('1', '5', '10')]
    rt = [paired(tm[ka], tm[kb])['ratio'] for ka, kb in keys if tm.get(ka) and tm.get(kb)]
    print(f"  median ratio {median(rt):.2f} over {len(rt)} of {len(keys)} cells, "
          f"range {min(rt):.2f} to {max(rt):.2f}")


def node_report():
    """Neural ODE comparisons (Section 5.3), paired by seed within the PyTorch sweep behind Table 3."""
    path = os.path.join(ROOT, 'results', 'runs_python.csv')
    rows = list(csv.DictReader(open(path)))
    c = defaultdict(dict)
    for r in rows:
        if r['sweep'] == '2026-10-05-node-final':
            c[(r['dataset'], r['basis'])][r['seed']] = float(r['test_agg'])
    family("NEURAL ODE, 2026-10-05-node-final: Legendre d=3 / static   [Section 5.3]",
           [(ds, c.get((ds, 'legendre')), c.get((ds, 'none'))) for ds in DS])
    family("NEURAL ODE: Legendre / monomial, d=3   [Section 5.3]",
           [(ds, c.get((ds, 'legendre')), c.get((ds, 'monomial'))) for ds in DS])
    family("NEURAL ODE: unscaled monomial / scaled monomial, d=3   [Section 5.3]",
           [(ds, c.get((ds, 'monomial-raw')), c.get((ds, 'monomial'))) for ds in DS])


if __name__ == '__main__':
    main()
