"""Paired comparisons of test error across seeds, shared by every generator.

A seed fixes the initialization and the minibatch order of every model trained
under it, so configurations run with the same seeds are compared seed by seed.
Seeds act as blocks, so pairing is valid even across architectures and
optimizers, where the starting functions differ; there it merely costs degrees
of freedom. Pairing adds power only where the within-seed correlation is high.
Errors are compared on the log scale: they are positive, their spread grows
with their size, and a ratio is what the paper states.

One criterion decides every claim, and a reader can apply it to the interval
printed beside it: a difference is claimed when the 95% interval
of the ratio excludes 1, and a ratio is within a margin of parity when the
whole interval lies inside it. No correction for multiple comparisons is made;
the paper reports each family's size and its median detectable ratio instead,
so the reader can judge how many exclusions chance would give. An exact
sign-flip test is reported as a sensitivity check.
"""
import math
import random
import statistics as st


def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta function."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 / (1 + aa * d if abs(1 + aa * d) > tiny else tiny)
        c = 1 + aa / c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 / (1 + aa * d if abs(1 + aa * d) > tiny else tiny)
        c = 1 + aa / c if abs(c) > tiny else tiny
        h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return h


def t_cdf(t, df):
    if t == 0:
        return 0.5
    x = df / (df + t * t)
    # For |t| below about 1e-8, x rounds to exactly 1 and the log(1-x) below
    # raises. The distribution is continuous there, so return the value at zero.
    if x >= 1.0:
        return 0.5
    lb = math.lgamma(df / 2 + 0.5) - math.lgamma(df / 2) - math.lgamma(0.5)
    front = math.exp(lb + (df / 2) * math.log(x) + 0.5 * math.log(1 - x))
    if x < (df / 2 + 1) / (df / 2 + 0.5 + 2):
        ib = front * _betacf(df / 2, 0.5, x) / (df / 2)
    else:
        ib = 1 - front * _betacf(0.5, df / 2, 1 - x) / 0.5
    return 1 - ib / 2 if t >= 0 else ib / 2


def t_quantile(p, df):
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def paired(a, b):
    """Compare two cells given as {seed: error}, over their shared seeds.

    Returns a dict: n, ratio (geometric mean of a/b), lo and hi (its two-sided
    95% interval), t, p (two-sided), and sd and se of the paired log
    differences, which mde() needs.
    """
    seeds = sorted(set(a) & set(b))
    diff = [math.log(a[s]) - math.log(b[s]) for s in seeds]
    n = len(diff)
    sd = st.stdev(diff)
    m, se = st.mean(diff), sd / math.sqrt(n)
    t = m / se if se > 0 else math.copysign(math.inf, m)
    q = t_quantile(0.975, n - 1)
    return dict(n=n, ratio=math.exp(m), lo=math.exp(m - q * se), hi=math.exp(m + q * se), t=t,
                p=2 * (1 - t_cdf(abs(t), n - 1)) if math.isfinite(t) else 0.0,
                sd=sd, se=se)


def mde(r, power=0.80, alpha=0.05):
    """Smallest ratio this comparison could have resolved, from a paired() result.

    The two-sided test at level alpha rejects when |t| > t_{1-alpha/2}; it does
    so with probability `power` when the true log ratio is at least
    (t_{1-alpha/2} + t_{power}) * se. Reporting this turns "no resolved
    difference" into a bounded claim: an effect below it would have been missed.
    The noncentral t is approximated by the central one, which slightly
    overstates the bound.
    """
    df = r['n'] - 1
    return math.exp((t_quantile(1 - alpha / 2, df) + t_quantile(power, df)) * r['se'])


def seed_gsd(cell):
    """Geometric standard deviation of one configuration across its seeds.

    The yardstick every effect is read against: a ratio smaller than this moves
    test error less than rerunning the same configuration with another seed.
    """
    return math.exp(st.stdev([math.log(v) for v in cell.values()]))


def differs(r):
    """A difference is claimed when the 95% interval of the ratio excludes 1."""
    return r['lo'] > 1 or r['hi'] < 1


def within(r, margin=0.10):
    """Within +-margin of parity: the whole 95% interval lies inside [1/(1+m), 1+m]."""
    return r['lo'] > 1 / (1 + margin) and r['hi'] < 1 + margin


def signflip_p(a, b, draws=20000):
    """Two-sided sign-flip permutation p for the mean paired log difference (exact up to 16 seeds)."""
    seeds = sorted(set(a) & set(b))
    diff = [math.log(a[s]) - math.log(b[s]) for s in seeds]
    obs = abs(sum(diff))
    n = len(diff)
    if n <= 16:
        hits = sum(1 for mask in range(1 << n)
                   if abs(sum(d if mask >> i & 1 else -d for i, d in enumerate(diff))) >= obs - 1e-12)
        return hits / (1 << n)
    rng = random.Random(0)
    hits = sum(1 for _ in range(draws)
               if abs(sum(d if rng.random() < 0.5 else -d for d in diff)) >= obs - 1e-12)
    return (hits + 1) / (draws + 1)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def page_trend(curves):
    """Page's L test for an ordered trend across treatments, seeds as blocks.

    curves: list over treatments (in order) of {seed: error}. Ranks within each
    seed, highest error = highest rank. Returns z (positive: error rises along
    the order) and the two-sided p from the normal approximation.
    """
    seeds = sorted(set.intersection(*(set(c) for c in curves)))
    k, n = len(curves), len(seeds)
    L = 0.0
    for s in seeds:
        vals = [c[s] for c in curves]
        order = sorted(range(k), key=lambda j: vals[j])
        for rank, j in enumerate(order, start=1):
            L += (j + 1) * rank
    mu = n * k * (k + 1) ** 2 / 4
    var = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    z = (L - mu) / math.sqrt(var)
    return z, 2 * norm_sf(abs(z))


def per_seed_slopes(curves, xs):
    """Slope of log error in x fitted per seed; mean slope as percent per unit x, with its t interval."""
    seeds = sorted(set.intersection(*(set(c) for c in curves)))
    xbar = st.mean(xs)
    sxx = sum((x - xbar) ** 2 for x in xs)
    slopes = []
    for s in seeds:
        y = [math.log(c[s]) for c in curves]
        ybar = st.mean(y)
        slopes.append(sum((x - xbar) * (v - ybar) for x, v in zip(xs, y)) / sxx)
    n = len(slopes)
    m, se = st.mean(slopes), st.stdev(slopes) / math.sqrt(n)
    q = t_quantile(0.975, n - 1)
    pct = lambda b: 100 * (math.exp(b) - 1)
    return dict(n=n, pct=pct(m), lo=pct(m - q * se), hi=pct(m + q * se),
                t=m / se, p=2 * (1 - t_cdf(abs(m / se), n - 1)))
