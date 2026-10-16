"""Tests for TimeParameterizedNet: basis correctness, Gram matrix, agreement
with the MATLAB implementation, gradient correctness, and speed.

Run:  python3 test_timeparam.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial import legendre as L

from timeparam import TimeParameterizedNet

TOL = 1e-5
_p, _f = 0, 0


def chk(name, ok, detail=""):
    global _p, _f
    if ok:
        _p += 1; print(f"  PASS  {name}")
    else:
        _f += 1; print(f"  FAIL  {name}   {detail}")


class Tiny(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.lin = nn.Linear(n, n)

    def forward(self, t, x):
        return torch.tanh(self.lin(x))


def ref_basis(basis, D, tau):
    if basis == "monomial":
        return np.array([tau ** k for k in range(D + 1)])
    return np.array([L.Legendre.basis(k)(tau) for k in range(D + 1)])


print("\n== 1. both bases live on [-1,1] and are T-independent ==")
for basis in ("monomial", "legendre"):
    conds = []
    for T in (1.0, 5.0, 25.0):
        tp = TimeParameterizedNet(Tiny(), tspan=[0.0, T], basis=basis, d=4)
        tt = torch.linspace(0, T, 12).reshape(-1, 1)
        A = tp._build_basis(tp._normalize_time(tt)).detach().numpy()
        conds.append(np.linalg.cond(A))
    chk(f"{basis}: endpoints map to tau = -1, +1",
        abs(tp._normalize_time(torch.tensor([[0.0]])).item() + 1) < 1e-12
        and abs(tp._normalize_time(torch.tensor([[T]])).item() - 1) < 1e-12)
    chk(f"{basis}: cond(A) = {conds[0]:.2f}, independent of T",
        max(conds) - min(conds) < 1e-5 * max(conds), f"{conds}")

print("\n== 2. cond matches the MATLAB implementation ==")
want = {"monomial": 7.16, "legendre": 2.29}
for basis, w in want.items():
    tp = TimeParameterizedNet(Tiny(), tspan=[0.0, 7.0], basis=basis, d=4)
    tt = torch.linspace(0, 7, 12).reshape(-1, 1)
    c = np.linalg.cond(tp._build_basis(tp._normalize_time(tt)).detach().numpy())
    chk(f"{basis}: cond(A) = {c:.2f} matches MATLAB {w}", abs(c - w) < 0.01, f"{c:.4f}")

print("\n== 3. weight interpolation equals sum_d p_d(tau) Theta_d ==")
torch.manual_seed(0)
for basis in ("monomial", "legendre"):
    T, D = 5.0, 3
    net = Tiny(2)
    tp = TimeParameterizedNet(net, tspan=[0.0, T], basis=basis, d=D + 1)
    W = tp.time_params["lin__weight"].detach().numpy()
    b = tp.time_params["lin__bias"].detach().numpy()
    worst = 0.0
    for t in (0.0, 1.3, T):
        tau = 2 * t / T - 1
        P = ref_basis(basis, D, tau)
        out = tp(torch.tensor(t), torch.eye(2)).detach().numpy()
        Wt = np.einsum("d,dij->ij", P, W)
        bt = np.einsum("d,di->i", P, b)
        worst = max(worst, np.abs(out - np.tanh(np.eye(2) @ Wt.T + bt)).max())
    chk(f"{basis}: theta(t) = sum_d p_d(tau) theta_d", worst < TOL, f"{worst:.2e}")

print("\n== 4. Gram matrix vs numerical quadrature on [-1,1] ==")
for basis in ("monomial", "legendre"):
    D = 4
    tp = TimeParameterizedNet(Tiny(), tspan=[0.0, 3.0], basis=basis, d=D + 1)
    M = tp.gram_matrix(dtype=torch.float64).numpy()
    x, w = np.polynomial.legendre.leggauss(60)
    P = np.stack([ref_basis(basis, D, xi) for xi in x])       # (nq, D+1)
    Mq = P.T @ (P * w[:, None])
    chk(f"{basis}: ||M - M_quad||/||M||", np.linalg.norm(M - Mq) / np.linalg.norm(Mq) < 1e-10,
        f"{np.linalg.norm(M - Mq) / np.linalg.norm(Mq):.2e}")
M_l = TimeParameterizedNet(Tiny(), tspan=[0, 1], basis="legendre", d=4).gram_matrix().numpy()
M_m = TimeParameterizedNet(Tiny(), tspan=[0, 1], basis="monomial", d=4).gram_matrix().numpy()
chk("Legendre Gram diagonal", np.abs(M_l - np.diag(np.diag(M_l))).max() < 1e-12)
chk("monomial Gram dense", np.abs(M_m - np.diag(np.diag(M_m))).max() > 1e-3)
print(f"     cond(M_leg) = {np.linalg.cond(M_l):.2f}   cond(M_mono) = {np.linalg.cond(M_m):.2f}")

print("\n== 5. BACKPROP: autograd vs finite differences ==")
for basis in ("monomial", "legendre"):
    torch.manual_seed(1)
    net = Tiny(3)
    tp = TimeParameterizedNet(net, tspan=[0.0, 4.0], basis=basis, d=4).double()
    x = torch.randn(6, 3, dtype=torch.float64)
    t = torch.tensor(1.7, dtype=torch.float64)
    target = torch.randn(6, 3, dtype=torch.float64)

    def loss_fn():
        return ((tp(t, x) - target) ** 2).sum()

    loss_fn().backward()
    train = [p for p in tp.parameters() if p.requires_grad]
    g_auto = torch.cat([p.grad.reshape(-1).clone() for p in train])
    flat = torch.cat([p.reshape(-1) for p in train]).detach()
    v = torch.randn_like(flat)
    h = 1e-6

    def perturb(eps):
        with torch.no_grad():
            i = 0
            for p in train:
                n = p.numel()
                p.copy_((flat[i:i + n] + eps * v[i:i + n]).reshape(p.shape))
                i += n
        return loss_fn().item()

    fd = (perturb(h) - perturb(-h)) / (2 * h)
    perturb(0.0)
    rel = abs(fd - (g_auto @ v).item()) / abs(fd)
    chk(f"{basis}: directional derivative matches FD", rel < 1e-6, f"rel err {rel:.2e}")

    tp.zero_grad()
    chk(f"{basis}: torch.autograd.gradcheck",
        torch.autograd.gradcheck(lambda *ps: tp(t, x).sum(),
                                 tuple(train),
                                 eps=1e-6, atol=1e-6, raise_exception=False))

print("\n== 6. batched call == per-sample loop (regression guard) ==")
torch.manual_seed(2)
net = Tiny(4)
tp = TimeParameterizedNet(net, tspan=[0.0, 10.0], basis="legendre", d=4).double()
x = torch.randn(32, 4, dtype=torch.float64)
t = torch.tensor(3.3, dtype=torch.float64)
fast = tp(t, x)
slow = torch.cat([tp(t, x[i:i + 1]) for i in range(x.shape[0])], dim=0)
chk("scalar-t batched path == one-sample-at-a-time",
    torch.allclose(fast, slow, atol=1e-12), f"{(fast - slow).abs().max():.2e}")
vec_t = tp(t.expand(32), x)
chk("vector-t path == scalar-t path", torch.allclose(fast, vec_t, atol=1e-12),
    f"{(fast - vec_t).abs().max():.2e}")

print("\n== 7. speed ==")
net = Tiny(50)
tp = TimeParameterizedNet(net, tspan=[0.0, 10.0], basis="legendre", d=4)
x = torch.randn(256, 50)
t = torch.tensor(2.0)
for _ in range(3):
    tp(t, x)
n = 50
t0 = time.perf_counter()
for _ in range(n):
    tp(t, x)
batched = (time.perf_counter() - t0) / n
t0 = time.perf_counter()
for _ in range(n):
    torch.cat([tp(t, x[i:i + 1]) for i in range(x.shape[0])], dim=0)
looped = (time.perf_counter() - t0) / n
print(f"     batched {batched*1e3:7.2f} ms/call   per-sample loop {looped*1e3:7.2f} ms/call"
      f"   speedup {looped/batched:.0f}x")
chk("batched path is faster", batched < looped)

print("\n== 8. TAYLOR TEST on the full objective, through the ODE solve ==")
# J(theta) = || readout(odeint(f_theta, embed(Y))) - C ||^2 / 2, differentiated
# w.r.t. the polynomial coefficients theta_d. Checks the whole backprop chain
# through the solver, not just the basis algebra.
#   |J(th+h v) - J(th)|                 = O(h)    -> halves as h halves
#   |J(th+h v) - J(th) - h <grad J, v>| = O(h^2)  -> quarters as h halves
from torchdiffeq import odeint

for basis in ("monomial", "legendre"):
    torch.manual_seed(3)
    T, D, nc = 2.0, 3, 4
    inner = Tiny(nc).double()
    func = TimeParameterizedNet(inner, tspan=[0.0, T], basis=basis, d=D + 1).double()
    embed = nn.Linear(3, nc).double()
    readout = nn.Linear(nc, 2).double()
    for m in (embed, readout):
        for q in m.parameters():
            q.requires_grad_(False)
    Y = torch.randn(5, 3, dtype=torch.float64)
    C = torch.randn(5, 2, dtype=torch.float64)
    tspan = torch.tensor([0.0, T], dtype=torch.float64)

    theta = [q for q in func.parameters() if q.requires_grad]
    flat0 = torch.cat([q.reshape(-1) for q in theta]).detach().clone()

    def set_flat(vec):
        with torch.no_grad():
            i = 0
            for q in theta:
                n = q.numel(); q.copy_(vec[i:i + n].reshape(q.shape)); i += n

    def J():
        zT = odeint(func, embed(Y), tspan, method="dopri5", rtol=1e-11, atol=1e-12)[-1]
        return 0.5 * ((readout(zT) - C) ** 2).sum()

    set_flat(flat0)
    for q in theta:
        if q.grad is not None: q.grad = None
    J0t = J(); J0t.backward()
    J0 = J0t.item()
    g = torch.cat([q.grad.reshape(-1) for q in theta])
    torch.manual_seed(4)
    v = torch.randn_like(flat0); v /= v.norm()
    gv = (g @ v).item()

    print(f"     {basis}:")
    print(f"     {'h':>10s} {'|J(th+hv)-J|':>14s} {'rate':>6s} {'|...-h<g,v>|':>14s} {'rate':>6s}")
    e0p = e1p = None
    r0s, r1s = [], []
    for h in [1e-1 * 0.5 ** k for k in range(6)]:
        set_flat(flat0 + h * v)
        Jh = J().item()
        e0 = abs(Jh - J0)
        e1 = abs(Jh - J0 - h * gv)
        r0 = e0p / e0 if e0p else float("nan")
        r1 = e1p / e1 if e1p else float("nan")
        if e0p: r0s.append(r0); r1s.append(r1)
        print(f"     {h:10.2e} {e0:14.4e} {r0:6.2f} {e1:14.4e} {r1:6.2f}")
        e0p, e1p = e0, e1
    set_flat(flat0)
    chk(f"{basis}: first-order residual decays at rate ~2",
        abs(np.median(r0s) - 2.0) < 0.15, f"median rate {np.median(r0s):.3f}")
    chk(f"{basis}: second-order residual decays at rate ~4 (gradient is correct)",
        abs(np.median(r1s) - 4.0) < 0.5, f"median rate {np.median(r1s):.3f}")

print("\n== 9. CONTINUOUS ADJOINT: gradients correct through the wrappers ==")
# torchdiffeq's odeint_adjoint takes gradients w.r.t. tuple(func.parameters()).
# TimeParameterizedNet keeps the wrapped net's original parameters registered
# with requires_grad=False, and the training script wraps func in an NFE
# counter. Either can make the adjoint return wrong or missing gradients.
# Check against backprop through the solver and by Taylor test.
from torchdiffeq import odeint as odeint_direct, odeint_adjoint

class NFECounter(nn.Module):
    """Mirrors CumulativeNFECounter in train_surrogate_node.py."""
    def __init__(self, net):
        super().__init__(); self.net = net; self.counter = 0
    def forward(self, t, y):
        self.counter += 1
        return self.net(t, y)

for basis in ("monomial", "legendre"):
    for wrapped in (False, True):
        torch.manual_seed(5)
        T, D, nc = 2.0, 3, 4
        func = TimeParameterizedNet(Tiny(nc).double(), tspan=[0.0, T],
                                    basis=basis, d=D + 1).double()
        if wrapped:
            func = NFECounter(func).double()
        embed = nn.Linear(3, nc).double(); readout = nn.Linear(nc, 2).double()
        for m in (embed, readout):
            for q in m.parameters():
                q.requires_grad_(False)
        Y = torch.randn(5, 3, dtype=torch.float64)
        Ct = torch.randn(5, 2, dtype=torch.float64)
        tspan = torch.tensor([0.0, T], dtype=torch.float64)
        theta = [q for q in func.parameters() if q.requires_grad]
        flat0 = torch.cat([q.reshape(-1) for q in theta]).detach().clone()

        def set_flat(vec):
            with torch.no_grad():
                i = 0
                for q in theta:
                    n = q.numel(); q.copy_(vec[i:i + n].reshape(q.shape)); i += n

        def loss(fn):
            zT = fn(func, embed(Y), tspan, method="dopri5", rtol=1e-11, atol=1e-12)[-1]
            return 0.5 * ((readout(zT) - Ct) ** 2).sum()

        def grad_of(fn):
            set_flat(flat0)
            for q in theta:
                q.grad = None
            L = loss(fn); L.backward()
            return L.item(), torch.cat([(q.grad if q.grad is not None
                                         else torch.zeros_like(q)).reshape(-1) for q in theta])

        tag = f"{basis}{' + NFE wrapper' if wrapped else ''}"
        J_d, g_d = grad_of(odeint_direct)
        J_a, g_a = grad_of(odeint_adjoint)
        chk(f"{tag}: adjoint returns a nonzero gradient for every coefficient",
            g_a.abs().min() > 0 and torch.isfinite(g_a).all(),
            f"{int((g_a == 0).sum())} zero entries")
        chk(f"{tag}: objective identical (adjoint vs direct)",
            abs(J_a - J_d) / abs(J_d) < 1e-10, f"{abs(J_a-J_d)/abs(J_d):.2e}")
        rel = (g_a - g_d).norm().item() / g_d.norm().item()
        # The continuous-adjoint (OTD) and backprop-through-solver (DTO)
        # gradients are different quantities that agree only to solver
        # tolerance (manuscript sect:optdisc). Assert closeness, not equality;
        # the Taylor test below certifies correctness.
        chk(f"{tag}: adjoint vs backprop gradient agree to solver tol (rel {rel:.2e})",
            rel < 1e-3, f"{rel:.3e}")

        if not wrapped:
            # Taylor test on the adjoint gradient itself
            torch.manual_seed(6)
            v = torch.randn_like(flat0); v /= v.norm()
            gv = (g_a @ v).item()
            set_flat(flat0); J0 = loss(odeint_adjoint).item()
            e1p, rates = None, []
            for h in [1e-1 * 0.5 ** k for k in range(5)]:
                set_flat(flat0 + h * v)
                e1 = abs(loss(odeint_adjoint).item() - J0 - h * gv)
                if e1p: rates.append(e1p / e1)
                e1p = e1
            set_flat(flat0)
            chk(f"{basis}: adjoint Taylor rate ~4 (median {np.median(rates):.2f})",
                abs(np.median(rates) - 4.0) < 0.5, f"median {np.median(rates):.3f}")

            # The OTD/DTO gap must shrink as the solver tightens, which
            # identifies it as discretization error rather than a bug in either
            # gradient.
            gaps = []
            for tol in (1e-5, 1e-7, 1e-9, 1e-11):
                set_flat(flat0)
                for q in theta: q.grad = None
                loss(lambda f, y, ts, **kw: odeint_direct(f, y, ts, method="dopri5",
                                                          rtol=tol, atol=tol * 1e-2)).backward()
                gd = torch.cat([q.grad.reshape(-1) for q in theta]).clone()
                for q in theta: q.grad = None
                loss(lambda f, y, ts, **kw: odeint_adjoint(f, y, ts, method="dopri5",
                                                           rtol=tol, atol=tol * 1e-2)).backward()
                ga = torch.cat([q.grad.reshape(-1) for q in theta]).clone()
                gaps.append((tol, (ga - gd).norm().item() / gd.norm().item()))
            set_flat(flat0)
            print("       OTD-vs-DTO gradient gap vs solver tolerance:")
            for tol, gp in gaps:
                print(f"         rtol={tol:.0e}  rel gap = {gp:.3e}")
            chk(f"{basis}: OTD/DTO gap shrinks as the solver tightens",
                gaps[-1][1] < gaps[0][1] / 10, f"{gaps[0][1]:.2e} -> {gaps[-1][1]:.2e}")

# ---------------------------------------------------------------------------
# monomial-raw: the unscaled control arm.
# It must (a) evaluate powers of raw t, not of tau; (b) leave the two scaled
# bases untouched; (c) reproduce the cond(A) values for T = 1, 5, 10 that the
# manuscript reports (section on weight parameterization and conditioning) and
# that the MATLAB implementation produces.
# ---------------------------------------------------------------------------
print("\n########## monomial-raw (unscaled control) ##########\n")

_net = Tiny(3)
_MATLAB_COND_A = {1.0: 102.01, 5.0: 349.48, 10.0: 1639.61}

for _T in (1.0, 5.0, 10.0):
    _tp = TimeParameterizedNet(_net, tspan=[0.0, _T], basis="monomial-raw", d=4)
    _t = torch.linspace(0, _T, 12).reshape(-1, 1)

    # (a) basis is powers of raw t
    _A = _tp._build_basis(_tp._normalize_time(_t)).detach().numpy()
    _ref = np.array([[float(tt) ** k for k in range(4)] for tt in _t.reshape(-1)])
    chk(f"monomial-raw T={_T:g}: A(j,k) = t_j^k on the raw interval",
        np.allclose(_A, _ref, rtol=1e-6, atol=1e-8),
        f"max dev {np.abs(_A - _ref).max():.2e}")

    # (c) cond(A) agrees with MATLAB
    _c = np.linalg.cond(_A)
    chk(f"monomial-raw T={_T:g}: cond(A) matches MATLAB ({_MATLAB_COND_A[_T]:.0f})",
        abs(_c - _MATLAB_COND_A[_T]) / _MATLAB_COND_A[_T] < 1e-3,
        f"got {_c:.2f}")

    # Gram matrix against direct quadrature of int_0^T t^d t^e dt, carried into
    # this module's convention by the (2/T) factor relating it to MATLAB's.
    _M = _tp.gram_matrix(dtype=torch.float64).numpy()
    _S = np.add.outer(np.arange(4), np.arange(4))
    _Mref = (2.0 / _T) * _T ** (_S + 1) / (_S + 1)
    chk(f"monomial-raw T={_T:g}: Gram matrix matches closed form",
        np.allclose(_M, _Mref, rtol=1e-9), f"max dev {np.abs(_M - _Mref).max():.2e}")

# (b) the scaled bases must be independent of T
for _b, _expect in (("monomial", 7.16), ("legendre", 2.29)):
    _conds = []
    for _T in (1.0, 5.0, 10.0):
        _tp = TimeParameterizedNet(_net, tspan=[0.0, _T], basis=_b, d=4)
        _t = torch.linspace(0, _T, 12).reshape(-1, 1)
        _conds.append(np.linalg.cond(
            _tp._build_basis(_tp._normalize_time(_t)).detach().numpy()))
    chk(f"{_b}: cond(A) is independent of T (scaling works)",
        max(_conds) - min(_conds) < 1e-6 and abs(_conds[0] - _expect) < 0.01,
        f"{[f'{c:.2f}' for c in _conds]}")

# The unscaled arm must actually differ from the scaled one, or it tests nothing.
_tp_raw = TimeParameterizedNet(_net, tspan=[0.0, 10.0], basis="monomial-raw", d=4)
_tp_scl = TimeParameterizedNet(_net, tspan=[0.0, 10.0], basis="monomial", d=4)
_t = torch.linspace(0, 10.0, 12).reshape(-1, 1)
chk("monomial-raw is genuinely distinct from monomial",
    not np.allclose(_tp_raw._build_basis(_tp_raw._normalize_time(_t)).detach().numpy(),
                    _tp_scl._build_basis(_tp_scl._normalize_time(_t)).detach().numpy()))

# Forward pass must run and stay finite for the unscaled arm.
_out = _tp_raw(torch.tensor(3.0), torch.randn(5, 3))
chk("monomial-raw: forward pass is finite",
    torch.isfinite(_out).all().item() and _out.shape == (5, 3))

# ---------------------------------------------------------------------------
# The regularizer of manuscript eq:regularizer. regularization_term() returns
# the integral in this module's tau-in-[-1,1] convention; the training script
# multiplies by the T/2 Jacobian to get int_0^T ||theta_NODE(t)||^2 dt. Check
# that composition against direct quadrature: a wrong Jacobian makes the MATLAB
# and Python pipelines minimize different objectives.
# ---------------------------------------------------------------------------
print("\n########## regularizer vs. direct quadrature ##########\n")

torch.manual_seed(0)
for _basis in ("monomial", "legendre", "monomial-raw"):
    for _T in (1.0, 5.0):
        _tp = TimeParameterizedNet(Tiny(3), tspan=[0.0, _T], basis=_basis, d=4)
        for _q in _tp.time_params.values():
            with torch.no_grad():
                _q.copy_(torch.randn_like(_q))
        _closed = (_T / 2.0) * _tp.regularization_term().item()

        _N = 200001
        _t = torch.linspace(0, _T, _N).reshape(-1, 1)
        _a = _tp._build_basis(_tp._normalize_time(_t))
        _tot = 0.0
        for _q in _tp.time_params.values():
            _P = _q.reshape(_q.shape[0], -1)
            _tot = _tot + ((_a @ _P) ** 2).sum(dim=1)
        _quad = torch.trapezoid(_tot, dx=_T / (_N - 1)).item()

        chk(f"{_basis} T={_T:g}: (T/2)*Gram equals int_0^T ||theta(t)||^2 dt",
            abs(_closed - _quad) <= 1e-5 * abs(_quad),
            f"gram {_closed:.6f} vs quad {_quad:.6f}")

print("\n" + "-" * 45)
print(f"{_p} passed, {_f} failed")
raise SystemExit(1 if _f else 0)
