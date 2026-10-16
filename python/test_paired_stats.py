#!/usr/bin/env python3
"""Checks for paired_stats.py. Standard library only, so it runs anywhere.

These assertions pin the identities that hold by construction, the two
distribution functions against published values, and `mde` and `seed_gsd`
against their definitions.

Usage: python3 test_paired_stats.py
"""
import math
import random
import statistics as st
import sys

from paired_stats import (differs, mde, paired, page_trend, seed_gsd, signflip_p,
                          t_cdf, t_quantile, within)

_p = _f = 0


def chk(name, ok, detail=''):
    global _p, _f
    if ok:
        _p += 1
        print(f'  PASS  {name}')
    else:
        _f += 1
        print(f'  FAIL  {name}  {detail}')


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


print('\n== 1. the t distribution ==')
# Published two-sided 5% critical values.
for df, want in ((1, 12.706), (9, 2.262), (30, 2.042), (120, 1.980)):
    got = t_quantile(0.975, df)
    chk(f't_quantile(0.975, {df}) = {want}', abs(got - want) < 5e-4, f'got {got:.5f}')
chk('t_cdf is the inverse of t_quantile',
    all(close(t_cdf(t_quantile(p, 9), 9), p, 1e-7) for p in (0.6, 0.9, 0.975, 0.999)))
chk('t_cdf is symmetric', close(t_cdf(-1.7, 12), 1 - t_cdf(1.7, 12), 1e-12))

print('\n== 2. paired() against a hand computation ==')
a = {'0': 1.0, '1': 2.0, '2': 4.0, '3': 8.0}
b = {'0': 1.0, '1': 1.0, '2': 1.0, '3': 1.0}
r = paired(a, b)
d = [0.0, math.log(2), math.log(4), math.log(8)]
chk('n counts shared seeds only', r['n'] == 4)
chk('ratio is the geometric mean of a/b', close(r['ratio'], math.exp(st.mean(d))))
chk('sd is the sample sd of the log differences', close(r['sd'], st.stdev(d)))
chk('se is sd / sqrt(n)', close(r['se'], st.stdev(d) / 2))
chk('t is mean / se', close(r['t'], st.mean(d) / r['se']))
chk('the 95% interval is symmetric on the log scale',
    close(math.log(r['hi'] / r['ratio']), math.log(r['ratio'] / r['lo'])))
chk('paired(b, a) inverts the ratio', close(paired(b, a)['ratio'], 1 / r['ratio']))
chk('seeds in one cell only are ignored',
    paired(dict(a, **{'9': 3.0}), b)['n'] == 4)

print('\n== 3. stored se agrees with the interval ==')
# mde() reads se directly; it must equal the se implied by the interval width.
for _ in range(20):
    rng = random.Random(_)
    x = {str(i): math.exp(rng.gauss(0, 0.3)) for i in range(10)}
    y = {str(i): math.exp(rng.gauss(0.1, 0.3)) for i in range(10)}
    q = paired(x, y)
    old = (math.log(q['hi']) - math.log(q['lo'])) / (2 * t_quantile(0.975, q['n'] - 1))
    if not close(old, q['se'], 1e-9):
        break
else:
    chk('interval-implied se equals the stored se on 20 random cells', True)

print('\n== 4. mde ==')
chk('mde matches its definition',
    close(mde(r), math.exp((t_quantile(0.975, 3) + t_quantile(0.80, 3)) * r['se'])))
chk('mde is above 1', mde(r) > 1)
chk('mde falls as the spread falls',
    mde(paired({k: v ** 0.1 for k, v in a.items()}, b)) < mde(r))
# Four times the seeds at the same spread should shrink the detectable effect
# by close to a factor of two on the log scale, plus the change in the quantiles.
wide = {str(i): math.exp(0.2 * ((-1) ** i)) for i in range(8)}
narrow = {str(i): 1.0 for i in range(8)}
few = {k: wide[k] for k in '0123'}, {k: narrow[k] for k in '0123'}
chk('more seeds resolve a smaller effect',
    mde(paired(wide, narrow)) < mde(paired(*few)))
# t_quantile(0.5, df) is zero up to the resolution of the bisection, which
# stops at the edge of the interval where t_cdf returns exactly 0.5.
chk('a power of 0.5 leaves only the alpha quantile',
    close(mde(r, power=0.5), math.exp(t_quantile(0.975, 3) * r['se']), 1e-6))

print('\n== 5. seed_gsd ==')
chk('seed_gsd matches its definition',
    close(seed_gsd(a), math.exp(st.stdev([math.log(v) for v in a.values()]))))
chk('a constant cell has no spread', close(seed_gsd({'0': 2.0, '1': 2.0, '2': 2.0}), 1.0))
chk('seed_gsd is scale free',
    close(seed_gsd(a), seed_gsd({k: 17 * v for k, v in a.items()})))

print('\n== 6. interval criterion ==')
same = {str(i): 1.0 + 1e-6 * i for i in range(10)}
ref = {str(i): 1.0 for i in range(10)}
chk('identical cells do not differ', not differs(paired(ref, ref)))
chk('identical cells are within 10%', within(paired(same, ref)))
wide = {str(i): math.exp(0.9 * ((-1) ** i)) for i in range(10)}
chk('a cell with a wide interval neither differs nor is within 10%',
    not differs(paired(wide, ref)) and not within(paired(wide, ref)))
shifted = {str(i): 1.5 * math.exp(0.01 * ((-1) ** i)) for i in range(10)}
chk('a cell shifted by 50% with little spread differs', differs(paired(shifted, ref)))
chk('differs agrees with p < 0.05',
    all(differs(paired(c, ref)) == (paired(c, ref)['p'] < 0.05) for c in (same, wide, shifted)))

print('\n== 7. sign-flip and Page ==')
chk('the sign-flip p is exact at ten seeds and bounded below by 2/1024',
    close(signflip_p({str(i): math.exp(0.5) for i in range(10)}, ref), 2 / 1024))
chk('the sign-flip p is 1 when every difference is zero',
    close(signflip_p(ref, ref), 1.0))
rising = [{str(s): float(k + 1) + 0.01 * s for s in range(6)} for k in range(4)]
z, pz = page_trend(rising)
chk('Page z is positive for a rising trend', z > 0, f'z={z:.2f}')
chk('Page p is small for a perfect trend', pz < 0.01, f'p={pz:.4f}')
zf, _ = page_trend(list(reversed(rising)))
chk('Page z flips sign with the order', close(zf, -z, 1e-9))

print('\n' + '-' * 45)
print(f'{_p} passed, {_f} failed')
sys.exit(1 if _f else 0)
