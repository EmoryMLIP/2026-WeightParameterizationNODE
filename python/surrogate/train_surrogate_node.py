"""
train_surrogate_node.py

Optimize-then-discretize neural ODE counterpart to the MATLAB
discretize-then-optimize pipeline (matlab/experiments/runExperiment.m), for
tables/table3.tex: ELM/CDR/DCR surrogate modeling via a continuous-depth ODE.

Architecture mirrors the MATLAB side's block1/block2/readout structure,
with the discrete nt-step ResNet dynamics replaced by an ODE integrated via
torchdiffeq:

    Y (params) --[Embed: Linear]--> z0
    z0 --[integrate dz/dt = f_theta(t,z), t in [0,T]]--> z(T)
    z(T) --[Readout: Linear]--> Chat (predicted targets)

f_theta is a plain MLP (no Hamiltonian structure). When --basis != none,
its weights are wrapped in TimeParameterizedNet (timeparam.py).

Usage (from python/):
    python3 surrogate/train_surrogate_node.py --dataset CDR --basis legendre \
        --degree 3 --seed 0 --results_json run.json
"""

import argparse
import builtins
import os
import random
import sys

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import dataset_config
from timeparam import TimeParameterizedNet


def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    return builtins.print(*args, **kwargs)


def load_ELM(data_dir):
    """Mirrors data/setupNNERDS.m. The split is the one stored in NNERDS.mat
    (idtrain/idval/idtest), identical to MATLAB's."""
    mat = sio.loadmat(os.path.join(data_dir, 'NNERDS.mat'))
    xall = mat['xall']              # (2486, 15)
    yall = mat['yall']              # (2486, 10)
    paramrange = mat['paramrange']  # (15, 2)
    # MATLAB indices are 1-indexed.
    idtrain = mat['idtrain'].flatten().astype(int) - 1
    idval = mat['idval'].flatten().astype(int) - 1
    idtest = mat['idtest'].flatten().astype(int) - 1

    pr_mean = paramrange.mean(axis=1)          # (15,)
    pr_range = paramrange[:, 1] - paramrange[:, 0]  # (15,)

    def norm_Y(Y):
        return (Y - pr_mean) * 2.0 / pr_range

    Ytrain, Yval, Ytest = norm_Y(xall[idtrain]), norm_Y(xall[idval]), norm_Y(xall[idtest])

    Ctrain_raw = yall[idtrain]
    minC = Ctrain_raw.min(axis=0)
    maxC = (Ctrain_raw - minC).max(axis=0)

    def norm_C(C):
        return (C - minC) / maxC

    Ctrain, Cval, Ctest = norm_C(Ctrain_raw), norm_C(yall[idval]), norm_C(yall[idtest])
    return Ytrain, Ctrain, Yval, Cval, Ytest, Ctest


def _random_split(Y, C, nTrain, nVal, data_dir=None, dataset=None):
    """Split into nTrain/nVal/rest.

    Uses the index permutation exported by matlab/analysis/exportSplits.m
    (data/splits.mat) so that the PyTorch and MATLAB pipelines use identical
    samples; MATLAB's randperm and NumPy's permutation differ for the same
    seed. Falls back to a fixed-seed NumPy permutation, with a warning, if
    splits.mat is missing."""
    n = Y.shape[0]
    if nTrain + nVal > n:
        raise ValueError(f"nTrain+nVal ({nTrain + nVal}) exceeds n ({n})")
    idx = None
    if data_dir is not None and dataset is not None:
        path = os.path.join(data_dir, 'splits.mat')
        if os.path.exists(path):
            rec = sio.loadmat(path, squeeze_me=True)
            if dataset in rec:
                entry = rec[dataset]
                idx = np.asarray(entry['idx'].item(), dtype=np.int64).ravel() - 1  # MATLAB is 1-based
                if idx.shape[0] != n:
                    raise ValueError(f"splits.mat[{dataset}] has n={idx.shape[0]}, data has n={n}")
    if idx is None:
        print(f"[WARNING] {dataset}: data/splits.mat not found -- falling back to a NumPy "
              f"permutation. This split will NOT match the MATLAB pipeline. Run "
              f"exportSplits.m in MATLAB to fix.")
        idx = np.random.RandomState(dataset_config.SPLIT_SEED).permutation(n)
    idxTrain, idxVal, idxTest = idx[:nTrain], idx[nTrain:nTrain + nVal], idx[nTrain + nVal:]
    return (Y[idxTrain], C[idxTrain], Y[idxVal], C[idxVal], Y[idxTest], C[idxTest])


def load_CDR(data_dir, cfg, input_scaling='zscore'):
    """Mirrors data/setupCDR.m for C (raw, no normalization).

    Y is z-scored with training statistics by default. setupCDR.m leaves Y raw,
    so the two pipelines train on differently scaled inputs unless
    input_scaling='raw' is passed.
    """
    mat = sio.loadmat(os.path.join(data_dir, 'CDR_Data.mat'))
    Y, C = mat['Y'].T, mat['C'].T   # (800,55) / (800,72) -- samples x features
    Yt, Ct, Yv, Cv, Yte, Cte = _random_split(Y, C, cfg['nTrain'], cfg['nVal'], data_dir, 'CDR')
    if input_scaling == 'raw':
        return Yt, Ct, Yv, Cv, Yte, Cte
    Ymean, Ystd = Yt.mean(axis=0), Yt.std(axis=0)
    return (Yt - Ymean) / Ystd, Ct, (Yv - Ymean) / Ystd, Cv, (Yte - Ymean) / Ystd, Cte


def load_DCR(data_dir, cfg, input_scaling='zscore'):
    """Targets are mean-centred with the training mean, as data/setupDCR.m does
    (Ct = C - meanTrain). Per-feature z-scoring would rescale each of the 882
    outputs independently and make relative errors incomparable with MATLAB's
    (tables/tableC1.tex). Under mean-centring the training-mean predictor
    scores exactly 1.0."""
    mat = sio.loadmat(os.path.join(data_dir, 'DCR_Data.mat'))
    Y, C = mat['Y'].T, mat['C'].T   # (10000,3) / (10000,882)
    Yt, Ct, Yv, Cv, Ytest, Ctest = _random_split(Y, C, cfg['nTrain'], cfg['nVal'], data_dir, 'DCR')
    Ymean, Ystd = Yt.mean(axis=0), Yt.std(axis=0)
    Cmean = Ct.mean(axis=0)          # mean-centre only, as in setupDCR.m
    if input_scaling == 'raw':
        return Yt, Ct - Cmean, Yv, Cv - Cmean, Ytest, Ctest - Cmean
    Yt_n, Yv_n, Ytest_n = (Yt - Ymean) / Ystd, (Yv - Ymean) / Ystd, (Ytest - Ymean) / Ystd
    return Yt_n, Ct - Cmean, Yv_n, Cv - Cmean, Ytest_n, Ctest - Cmean


LOADERS = {'ELM': load_ELM, 'CDR': load_CDR, 'DCR': load_DCR}


# ---------------------------------------------------------------------------
# Model: plain-MLP dynamics, continuous-depth ODE via torchdiffeq, optionally
# time-parameterized via TimeParameterizedNet.
# ---------------------------------------------------------------------------

class ODEFunc(nn.Module):
    """Vector field dz/dt = f(z). Ignores t directly; time dependence, when
    present, comes from TimeParameterizedNet varying this net's weights.

    arch='mlp'          nc -> hidden -> nc with tanh between.
    arch='single_layer' tanh(K z + b), K in R^{nc x nc}: the MATLAB ResNN dynamics
                            block2 = ResNN(singleLayer(dense([nc,nc]),'Bin',eye(nc)),...)
                        (activation defaults to tanh). With --method euler,
                        --n_steps 12 and --backprop-through-solver it is the same
                        model as the MATLAB pipeline.
    """

    def __init__(self, nc, hidden, arch='mlp'):
        super().__init__()
        if arch == 'single_layer':
            self.net = nn.Sequential(nn.Linear(nc, nc), nn.Tanh())
        else:
            self.net = nn.Sequential(
                nn.Linear(nc, hidden),
                nn.Tanh(),
                nn.Linear(hidden, nc),
            )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, z):
        return self.net(z)


class SurrogateODENet(nn.Module):
    """Embed -> integrate dz/dt=f_theta(t,z), t in [0,T] -> Readout."""

    FIXED_STEP = ('euler', 'rk4', 'midpoint')

    def __init__(self, input_dim, output_dim, nc, hidden, T, basis, degree,
                 odeint_fn, method, rtol, atol, n_steps=12, arch='mlp'):
        super().__init__()
        # single_layer reproduces MATLAB's block1
        #     NN({singleLayer(dense([nc,nY],'Bin',eye(nc)))})
        # i.e. tanh(K_in y) with no bias, because 'Bin' is an argument to
        # dense(), not to singleLayer(). The two models then have the same
        # parameter count.
        self.embed = (nn.Sequential(nn.Linear(input_dim, nc, bias=False), nn.Tanh())
                      if arch == 'single_layer' else nn.Linear(input_dim, nc))
        func = ODEFunc(nc, hidden, arch)
        if basis == 'none':
            self.func = func
        else:
            self.func = TimeParameterizedNet(func, tspan=[0.0, T], basis=basis, d=degree+1)
        self.readout = nn.Linear(nc, output_dim)
        # Handle to the theta_NODE module for the regularizer. Kept separately
        # because training rebinds self.func to the NFE counter wrapper.
        self._theta_node = self.func
        self.T = T
        self.odeint_fn = odeint_fn
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.n_steps = n_steps

    def _solve(self, func, z0):
        t_span = torch.tensor([0.0, self.T], device=z0.device, dtype=z0.dtype)
        if self.method in self.FIXED_STEP:
            # Fixed-step Euler u_{j+1} = u_j + h f(t_j, u_j), h = T/n_steps, is the
            # MATLAB ResNet discretization when n_steps = nt = 12.
            return self.odeint_fn(func, z0, t_span, method=self.method,
                                  options={'step_size': self.T / self.n_steps})[-1]
        return self.odeint_fn(func, z0, t_span, method=self.method,
                              rtol=self.rtol, atol=self.atol)[-1]

    def warm_start_readout(self, Y, C, alpha2):
        r"""Initialize the readout with the regularized least-squares solution.

        Mirrors the 'sgd' branch of matlab/experiments/runExperiment.m:

            YN0 = forwardProp(net,th0,Yt);  X0 = [YN0; ones(1,n)];
            W   = (Ct*X0') / (X0*X0' + alpha2*eye(size(X0,1)));

        The readout is linear in the latent state, so its optimum given the
        current dynamics has a closed form. This matters most when the readout
        is large: on DCR it is 16 -> 882 and holds most of the parameters.

        This is not variable projection: the readout stays a trainable parameter
        and ADAM keeps updating it.
        """
        with torch.no_grad():
            z0 = self.embed(Y)
            zT = self._solve(self.func, z0)
            A = torch.cat([zT, torch.ones(zT.shape[0], 1, dtype=zT.dtype, device=zT.device)], 1)
            A, Cd = A.double(), C.double()
            G = A.t() @ A + alpha2 * torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
            W = torch.linalg.solve(G, A.t() @ Cd)          # (nc+1, nOut)
            self.readout.weight.copy_(W[:-1, :].t().float())
            self.readout.bias.copy_(W[-1, :].float())

    def regularization(self):
        r"""R(theta)/alpha from manuscript eq:regularizer:

            R(theta) = (alpha/2)||theta_static||^2
                     + (alpha/2) \int_0^T ||theta_NODE(t)||^2 dt

        Returns everything except the alpha factor, so the caller writes
        loss = misfit + alpha * model.regularization().

        theta_static is (K_in, b_in, K_out, b_out); theta_NODE is the dynamics.
        For a parameterized net the time integral goes through the Gram matrix
        (timeparam.regularization_term), which is expressed on tau in [-1,1];
        the T/2 Jacobian carries it back to \int_0^T. For the static-weight
        neural ODE theta_NODE(t) is constant, so the integral is T||theta_0||^2.
        """
        static = sum((p ** 2).sum()
                     for m in (self.embed, self.readout) for p in m.parameters())
        src = self._theta_node
        if isinstance(src, TimeParameterizedNet):
            node = (self.T / 2.0) * src.regularization_term()
        else:
            node = self.T * sum((p ** 2).sum() for p in src.parameters())
        return 0.5 * (static + node)

    def forward(self, Y, func_override=None):
        func = func_override if func_override is not None else self.func
        z0 = self.embed(Y)
        return self.readout(self._solve(func, z0))


class CumulativeNFECounter(nn.Module):
    """Wraps the ODE dynamics function and counts every forward call across the
    whole training run (never reset), unlike ProgressWrapperNet, which counts
    one eval_region() call. Gives the loss-vs-cumulative-NFE curves of
    fig:neuralODE_loss. Wrap model.func with this before training so that
    training-batch and validation passes both count."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.counter = 0

    def forward(self, t, y):
        self.counter += 1
        return self.net(t, y)


class ProgressWrapperNet(nn.Module):
    """Counts ODE function calls, for NFE reporting."""

    def __init__(self, net, desc="Evaluating ODE"):
        super().__init__()
        self.net = net
        self.counter = 0
        self.desc = desc
        self.pbar = tqdm(desc=self.desc, unit=" calls", file=sys.stdout, dynamic_ncols=True)

    def forward(self, t, y):
        self.counter += 1
        return self.net(t, y)

    def close(self):
        self.pbar.set_description(f"{self.desc} (done, {self.counter} calls)")
        self.pbar.close()


def relative_error(pred, true, eps=1e-12):
    """||pred-true||_2 / ||true||_2, the convention of MATLAB's
    relErrTrain/Val/Test."""
    num = torch.sqrt(torch.sum((pred - true) ** 2))
    den = torch.sqrt(torch.sum(true ** 2)) + eps
    return (num / den).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ELM', 'CDR', 'DCR'], required=True)
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'),
                         help='Directory containing NNERDS.mat / CDR_Data.mat / DCR_Data.mat')
    parser.add_argument('--method', type=str, default='dopri5',
                        choices=['dopri5', 'euler', 'rk4', 'midpoint'],
                        help="Adaptive 'dopri5' is the paper's neural ODE. 'euler' with "
                             "--n_steps 12 is the MATLAB ResNet discretization, so with "
                             "--backprop-through-solver it reproduces "
                             "discretize-then-optimize in this code path.")
    parser.add_argument('--input_scaling', choices=['zscore', 'raw'], default='zscore',
                        help="How Y is scaled. 'zscore' uses training statistics; 'raw' leaves Y "
                             "untouched, as data/setup{CDR,DCR}.m do. Use 'raw' to compare "
                             "against the MATLAB pipeline.")
    parser.add_argument('--arch', choices=['mlp', 'single_layer'], default='mlp',
                        help="'single_layer' reproduces the MATLAB half's ResNN dynamics "
                             "tanh(K z + b) and its tanh input block. With --method euler "
                             "--n_steps 12 --backprop-through-solver the two implementations "
                             "are the same model.")
    parser.add_argument('--n_steps', type=int, default=12,
                        help='Number of fixed steps when --method is euler/rk4/midpoint.')
    # The neural ODE side is optimize-then-discretize (manuscript sect:optdisc),
    # so odeint_adjoint is the default. --backprop-through-solver selects the
    # discretize-then-optimize gradient (torchdiffeq.odeint) for direct comparison.
    parser.add_argument('--backprop-through-solver', dest='backprop_through_solver',
                        action='store_true',
                        help='Differentiate through the solver steps (discretize-then-optimize) '
                             'instead of using the continuous adjoint (the default).')
    parser.add_argument('--rtol', type=float, default=1e-7)
    parser.add_argument('--atol', type=float, default=1e-9)

    parser.add_argument('--basis', choices=['legendre', 'monomial', 'monomial-raw', 'none'],
                        default='legendre',
                        help="'monomial-raw' is the unscaled control: the monomial basis on the "
                             "raw interval [0, T] rather than on tau in [-1, 1].")
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--nc', type=int, default=None,
                        help='Embedding / dynamics width. Default: per-dataset value from '
                             'dataset_config.py, shared with MATLAB getDatasetConfig.m.')
    parser.add_argument('--hidden', type=int, default=50, help='ODEFunc hidden width')
    parser.add_argument('--T', type=float, default=1.0, help='Integration horizon')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Minibatch size. 0 (default) derives it as '
                             'nTrain/minibatches_per_epoch, so every dataset sees the same number '
                             'of epochs for a given iteration count.')
    parser.add_argument('--warm_start_readout', type=int, default=1,
                        help='Initialize the readout with the regularized least-squares solution '
                             'before training, as runExperiment.m does for its ADAM arm. Set 0 to '
                             'start from a random readout.')
    parser.add_argument('--minibatches_per_epoch', type=int, default=20,
                        help='Used only when --batch_size 0.')
    parser.add_argument('--lr', type=float, default=1e-2,
                         help='ADAM/RMSprop learning rate.')
    parser.add_argument('--optimizer', choices=['rmsprop', 'adam'], default='adam',
                         help='Optimizer.')
    parser.add_argument('--test_freq', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--lr_decay_every', type=int, default=0)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1e-10,
                        help='Regularization strength in eq:regularizer, applied to both the static '
                             'weights and the L2(0,T) norm of theta_NODE. Default matches the '
                             'MATLAB arm (alpha1 = alpha2 = 1e-10), selected on validation '
                             'error (appendix Selected settings). Set 0 to disable.')

    parser.add_argument('--results_json', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.backprop_through_solver:
        from torchdiffeq import odeint as odeint_fn
        print("Gradients: backprop through solver steps (discretize-then-optimize).")
    else:
        from torchdiffeq import odeint_adjoint as odeint_fn
        print("Gradients: continuous adjoint (optimize-then-discretize).")

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(f"The device being used is {device}.")
    print(f"Dataset: {args.dataset} | Seed: {args.seed} | Basis: {args.basis} | "
          f"Degree: {args.degree} | T: {args.T}")

    cfg = dataset_config.get(args.dataset)
    if args.nc is None:
        args.nc = cfg['nc']          # per-dataset width, shared with MATLAB
    if args.dataset == 'ELM':
        if args.input_scaling != 'zscore':
            raise SystemExit("--input_scaling only applies to CDR and DCR; ELM ships its own split")
        Ytr, Ctr, Yv, Cv, Yte, Cte = LOADERS[args.dataset](args.data_dir)
    else:
        Ytr, Ctr, Yv, Cv, Yte, Cte = LOADERS[args.dataset](args.data_dir, cfg,
                                                           input_scaling=args.input_scaling)
    print(f"Split sizes: train={Ytr.shape[0]} val={Yv.shape[0]} test={Yte.shape[0]} "
          f"| input_dim={Ytr.shape[1]} output_dim={Ctr.shape[1]} | nc={args.nc}")

    # Relative error of the training-mean predictor, as a baseline for every
    # reported error (matches matlab/lib/trivialPredictorError.m).
    _mu = Ctr.mean(axis=0, keepdims=True)
    trivial = {k: float(np.linalg.norm(np.broadcast_to(_mu, X.shape) - X) / np.linalg.norm(X))
               for k, X in (('train', Ctr), ('val', Cv), ('test', Cte))}
    print(f"Trivial (mean) predictor relative error: "
          f"train {trivial['train']:.4f} val {trivial['val']:.4f} test {trivial['test']:.4f}")

    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Ytr, Ctr, Yv, Cv, Yte, Cte = to_t(Ytr), to_t(Ctr), to_t(Yv), to_t(Cv), to_t(Yte), to_t(Cte)

    if args.basis == 'none':
        args.degree = 0  # placeholder, as in MATLAB for basis='none'
        print("Using STATIC (identity) baseline -- no time-parameterization.")

    model = SurrogateODENet(
        input_dim=Ytr.shape[1], output_dim=Ctr.shape[1], nc=args.nc, hidden=args.hidden,
        T=args.T, basis=args.basis, degree=args.degree, odeint_fn=odeint_fn, n_steps=args.n_steps,
        arch=args.arch,
        method=args.method, rtol=args.rtol, atol=args.atol,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params}")

    # Count every forward call (training batches and validation checks) in one
    # running total for the loss-vs-NFE trace of fig:neuralODE_loss.
    nfe_counter = CumulativeNFECounter(model.func)
    model.func = nfe_counter
    nfe_trace = []  # list of (iter, cumulative_nfe, train_relerr_batch, val_relerr)

    if args.warm_start_readout:
        model.warm_start_readout(Ytr, Ctr, args.alpha)
        with torch.no_grad():
            print(f"Readout warm-started by regularized least squares "
                  f"(train relerr at init: {relative_error(model(Ytr), Ctr):.4f})")

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_decay_every > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every,
                                               gamma=args.lr_decay_gamma)

    n_train = Ytr.shape[0]
    if args.batch_size <= 0:
        args.batch_size = max(1, round(n_train / args.minibatches_per_epoch))
    epochs = args.niters * args.batch_size / n_train
    print(f"Batch size: {args.batch_size} ({args.minibatches_per_epoch} minibatches/epoch) | "
          f"{args.niters} iterations = {epochs:.0f} epochs")

    best_val_relerr = float('inf')
    best_state_dict = None
    best_iter = None
    no_improve_checks = 0
    stopped_early = False
    stop_iter = None

    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad()
        idx = np.random.randint(0, n_train, size=min(args.batch_size, n_train))
        Yb, Cb = Ytr[idx], Ctr[idx]
        Cb_hat = model(Yb)
        # Per-sample terminal misfit (1/2)||h(u(T)) - c||^2 averaged over the
        # batch (manuscript eq:misfit), the objective the MATLAB half minimizes.
        loss = 0.5 * ((Cb_hat - Cb) ** 2).sum() / Yb.shape[0]
        if args.alpha > 0:
            loss = loss + args.alpha * model.regularization()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if itr % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                Cv_hat = model(Yv)
                val_relerr = relative_error(Cv_hat, Cv)
                Ctr_hat_batch = model(Yb)
                train_relerr_batch = relative_error(Ctr_hat_batch, Cb)
            cumulative_nfe = nfe_counter.counter
            nfe_trace.append({'iter': itr, 'nfe': cumulative_nfe,
                               'train_relerr': train_relerr_batch, 'val_relerr': val_relerr})
            print(f"Iter {itr:04d} | NFE {cumulative_nfe} | Val RelErr: {val_relerr:.6f}")

            if val_relerr < best_val_relerr:
                best_val_relerr = val_relerr
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                best_iter = itr
                no_improve_checks = 0
            else:
                no_improve_checks += 1
                if args.early_stop_patience > 0:
                    print(f"  No improvement ({no_improve_checks}/{args.early_stop_patience} checks since last best)")
                    if no_improve_checks >= args.early_stop_patience:
                        stopped_early = True
                        stop_iter = itr
                        print(f"\nEarly stopping triggered at iter {itr} "
                              f"(no improvement for {args.early_stop_patience} consecutive val checks). "
                              f"Reporting best checkpoint from iter {best_iter}.")
                        break

    if best_state_dict is None:
        best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        best_iter = args.niters
    model.load_state_dict(best_state_dict)
    model.eval()

    def eval_region(Y, C, desc):
        wrapped = ProgressWrapperNet(model.func, desc=desc)
        with torch.no_grad():
            relerr = relative_error(model(Y, func_override=wrapped), C)
        nfe = wrapped.counter
        wrapped.close()
        return relerr, nfe

    train_relerr, train_nfe = eval_region(Ytr, Ctr, "Evaluating TRAIN region")
    val_relerr, val_nfe = eval_region(Yv, Cv, "Evaluating VAL region")
    test_relerr, test_nfe = eval_region(Yte, Cte, "Evaluating TEST region (held out)")

    print(f"\nTraining complete (stopped_early={stopped_early}, best_iter={best_iter}).")
    print(f"  TRAIN relerr: {train_relerr:.6f} (NFE: {train_nfe})")
    print(f"  VAL   relerr: {val_relerr:.6f} (NFE: {val_nfe})  [selection criterion]")
    print(f"  TEST  relerr: {test_relerr:.6f} (NFE: {test_nfe})  [held out, reported once]")

    summary = {
        'dataset': args.dataset,
        'basis': args.basis,
        'nc': args.nc,
        'gradient': 'backprop_through_solver' if args.backprop_through_solver else 'continuous_adjoint',
        'trivial_relerr': trivial,
        'split_seed': dataset_config.SPLIT_SEED,
        'degree': args.degree,
        'seed': args.seed,
        'nc': args.nc,
        'hidden': args.hidden,
        'T': args.T,
        'niters_requested': args.niters,
        'batch_size': args.batch_size,
        'minibatches_per_epoch': args.minibatches_per_epoch,
        'arch': args.arch,
        'input_scaling': args.input_scaling,
        'method': args.method,
        'n_steps': args.n_steps,
        'warm_start_readout': bool(args.warm_start_readout),
        'epochs': args.niters * args.batch_size / Ytr.shape[0],
        'grad_clip': args.grad_clip,
        'lr': args.lr,
        'lr_decay_every': args.lr_decay_every,
        'lr_decay_gamma': args.lr_decay_gamma,
        'early_stop_patience': args.early_stop_patience,
        'alpha': args.alpha,
        'loss': 'least_squares',
        'rtol': args.rtol,
        'atol': args.atol,
        'stopped_early': stopped_early,
        'stop_iter': stop_iter,
        'best_iter': best_iter,
        'trainable_params': num_params,
        # {train,val,test}_relerr all use the best-on-validation checkpoint.
        'train_relerr': train_relerr,
        'val_relerr': val_relerr,
        'test_relerr': test_relerr,
        'train_nfe': train_nfe,
        'val_nfe': val_nfe,
        'test_nfe': test_nfe,
        'nfe_trace': nfe_trace,
    }
    results_path = args.results_json or f"results_{args.dataset}_{args.basis}_d{args.degree}_seed{args.seed}.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {results_path}")


if __name__ == '__main__':
    main()
