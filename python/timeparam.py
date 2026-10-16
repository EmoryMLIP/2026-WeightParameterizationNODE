import torch
import torch.nn as nn
import torch.nn.functional as F


def sanitize_key(key: str) -> str:
    """Replace '.' with '__' to avoid conflicts in ParameterDict keys."""
    return key.replace('.', '__')


def unsanitize_key(skey: str) -> str:
    """Revert the key sanitization so we can load into net via load_state_dict."""
    return skey.replace('__', '.')


class TimeParameterizedNet(nn.Module):
    """
    Wrap a PyTorch module (net) so that its weights become time-dependent
    via a polynomial expansion of degree d-1 in time.

    - Each original parameter net.state_dict()[name] is replaced by a new parameter
      of shape (d, *original_param.shape).
    - In the forward pass, the basis [p_0, ..., p_(d-1)] is evaluated at t and the
      d coefficient tensors are summed with those weights to give the parameter at t.
    """

    def __init__(self, net: nn.Module, tspan, basis="legendre", d: int = 3):
        """
        Args:
            net   (nn.Module): The original network whose parameters we wrap.
            tspan (list or tuple of float): [t0, t1], the absolute time span over
                         which this net will be evaluated. Incoming times are
                         rescaled onto [-1, 1] using this span.
            d     (int): The degree of the polynomial expansion, i.e. we learn
                         coefficients for [1, t, ..., t^(d-1)].
            basis (str): "monomial", "legendre", or "monomial-raw"
                         (the unscaled control; see _normalize_time).
        """
        super().__init__()
        self.net = net
        self.tspan = tspan
        self.t0 = float(tspan[0])
        self.t1 = float(tspan[1])
        if self.t1 <= self.t0:
            raise ValueError(f"tspan must satisfy t1 > t0, got tspan={tspan}")
        self.d = d
        self.basis = basis.lower()

        for param in self.net.parameters():
            param.requires_grad = False

        original_state = self.net.state_dict()

        # One parameter of shape (d, *orig_shape) per entry of the wrapped net.
        self.time_params = nn.ParameterDict()

        # ParameterDict keys cannot contain '.', so map sanitized key -> original
        # key to load the evaluated weights back into net.
        self.key_mapping = {}

        for name, orig_tensor in original_state.items():
            p = nn.Parameter(torch.zeros(d, *orig_tensor.shape))

            # The constant coefficient starts at the original weights.
            with torch.no_grad():
                p[0].copy_(orig_tensor)

            sname = sanitize_key(name)
            self.time_params[sname] = p
            self.key_mapping[sname] = name

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale absolute time t in [t0, t1] onto tau in [-1, 1], clamped.

        Both bases are evaluated on this same tau (manuscript eq:tau,
        matlab/meganet/getPolynomialBasis.m), so monomial against Legendre
        compares basis functions rather than domains. Evaluating the monomial
        basis on [0, 1] instead would inflate cond(A) and turn its Gram matrix
        into the Hilbert matrix.
        """
        if self.basis == "monomial-raw":
            # Unscaled control: p_d(t) = t^d on the raw interval [t0, t1].
            # V[0,T] = V[0,1] diag(1, T, ..., T^D) (manuscript eq:column_scaling),
            # so cond(V) can grow like T^D through the choice of time unit alone.
            # Comparing with "monomial" isolates time scaling from orthogonality.
            return t.clamp(self.t0, self.t1)
        tau = 2.0 * (t - self.t0) / (self.t1 - self.t0) - 1.0
        return tau.clamp(-1.0, 1.0)

    def _build_basis(self, tau: torch.Tensor) -> torch.Tensor:
        """(batch, d) basis matrix from normalized time tau in [-1, 1]."""
        if self.basis in ("monomial", "monomial-raw"):
            a = torch.cat([tau ** i for i in range(self.d)], dim=1)
        elif self.basis == "legendre":
            a_list = [torch.ones_like(tau)]          # P0 = 1
            if self.d > 1:
                a_list.append(tau)                   # P1 = tau
            for n in range(2, self.d):
                Pn = ((2 * n - 1) * tau * a_list[-1] - (n - 1) * a_list[-2]) / n
                a_list.append(Pn)
            a = torch.cat(a_list, dim=1)
        else:
            raise ValueError(f"Unknown basis type: {self.basis}")
        return a

    def gram_matrix(self, device=None, dtype=None) -> torch.Tensor:
        """M_de := int_{-1}^{1} p_d(tau) p_e(tau) dtau, the Gram matrix M of
        manuscript eq:grad_coefficients on the normalized coordinate.

        Closed form (matlab/meganet/getPolynomialBasis.m returns the same matrix
        scaled by the T/2 Jacobian):
          monomial: M_de = 2/(d+e+1) if d+e even else 0      (not diagonal)
          legendre: M_de = delta_de * 2/(2d+1)               (diagonal)
        A diagonal M decouples the regularization gradients across degrees.
        """
        d = self.d
        dt = dtype or torch.float32
        idx = torch.arange(d, dtype=dt, device=device)
        if self.basis == "monomial":
            S = idx.unsqueeze(0) + idx.unsqueeze(1)
            M = torch.where(S % 2 == 0, 2.0 / (S + 1.0), torch.zeros_like(S))
        elif self.basis == "monomial-raw":
            # int_0^T t^d t^e dt = T^(S+1)/(S+1), times the (2/T) factor that
            # relates every entry here to getPolynomialBasis.m, gives
            # 2 T^S/(S+1). Dense and badly scaled; the Hilbert matrix at T = 1.
            S = idx.unsqueeze(0) + idx.unsqueeze(1)
            T = self.t1 - self.t0
            M = 2.0 * (T ** S) / (S + 1.0)
        elif self.basis == "legendre":
            M = torch.diag(2.0 / (2.0 * idx + 1.0))
        else:
            raise ValueError(f"Unknown basis type: {self.basis}")
        return M

    def regularization_term(self) -> torch.Tensor:
        """
        Returns sum_{d,e} M_de <theta_d, theta_e> with M from gram_matrix(),
        i.e. \\int_{-1}^{1} ||theta_NODE(tau)||_2^2 dtau on the normalized
        coordinate. The caller multiplies by the Jacobian T/2 to obtain
        \\int_0^T ||theta_NODE(t)||_2^2 dt, the L^2(0,T) term of
        manuscript eq:regularizer (see train_surrogate_node.py;
        test_timeparam.py checks the composition against quadrature), and by
        alpha/2. Autograd differentiates this scalar directly; its gradient is the
        regularization term of manuscript eq:grad_coefficients.
        """
        total = None
        for time_param in self.time_params.values():
            # time_param: (d, *orig_shape) -> flatten to (d, N)
            d = time_param.shape[0]
            P = time_param.reshape(d, -1)
            M = self.gram_matrix(device=P.device, dtype=P.dtype)
            term = torch.sum(P * (M @ P))  # trace(P^T M P)
            total = term if total is None else total + term
        return total

    def forward(self, t, x):
        """
        Evaluate the weights at t as sum_i a[i] * time_params[name][i], with a the
        basis row at the normalized time, and apply the wrapped net.

        Args:
            t (float or Tensor): Absolute time, in the units of tspan.
            x (Tensor): Input features.

        Returns:
            Tensor: Output of the wrapped net at time t.
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)

        # torchdiffeq calls f(t, y) with a scalar t, so the whole batch shares
        # one basis row and one parameter set.
        if t.dim() == 0:
            tau = self._normalize_time(t.reshape(1, 1))
            a = self._build_basis(tau)[0]                       # (d,)
            params = {self.key_mapping[sname]: torch.einsum('d,d...->...', a, tp)
                      for sname, tp in self.time_params.items()}
            return torch.func.functional_call(self.net, params, (t, x))

        # Per-sample times (not used by torchdiffeq). Loops because each sample
        # needs its own parameter set.
        tt = t.reshape(-1, 1)
        if tt.shape[0] != x.shape[0]:
            tt = tt.expand(x.shape[0], -1)
        a = self._build_basis(self._normalize_time(tt))          # (batch, d)

        def one(a_i, t_i, x_i):
            params = {self.key_mapping[sname]: torch.einsum('d,d...->...', a_i, tp)
                      for sname, tp in self.time_params.items()}
            return torch.func.functional_call(self.net, params, (t_i, x_i.unsqueeze(0)))

        return torch.cat([one(a[i], tt[i, 0], x[i]) for i in range(x.shape[0])], dim=0)


# Example usage
if __name__ == "__main__":
    import torch.optim as optim

    # A base net with the (t, x) call signature.
    class BaseNet(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=16, output_dim=1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, t, x):
            out = self.fc1(x)
            out = F.relu(out)
            out = self.fc2(out)
            return out

    base_net = BaseNet(input_dim=2, hidden_dim=4, output_dim=1)

    t0, t1 = 0.0, 1.0
    time_net = TimeParameterizedNet(base_net, tspan=[t0, t1], d=5)

    # Print the time_params and shapes
    for name, param in time_net.time_params.items():
        print(f"{name} : {param.shape}")

    # Example: forward pass at different times
    x = torch.randn(5, 2)  # batch of 5
    y_t0 = time_net(0.0, x)  # output at time = 0.0
    y_t1 = time_net(1.0, x)  # output at time = 1.0
    y_test = base_net(0.0, x)

    print("Output at t=0.0 :", y_t0.squeeze())
    print("Output at t=1.0 :", y_t1.squeeze())
    print("Output base net :", y_test.squeeze())

    # Show difference in the actual parameters at t=0 vs t=1
    sd_0 = {}
    sd_1 = {}
    with torch.no_grad():
        time_net(0.0, x)  # loads net with the parameters at t=0
        sd_0 = {k: p.clone() for k, p in time_net.net.state_dict().items()}

        time_net(1.0, x)  # loads net with the parameters at t=1
        sd_1 = {k: p.clone() for k, p in time_net.net.state_dict().items()}

    print("\nComparing final layer (fc2) weight at t=0 vs t=1:")
    print("fc2.weight at t=0:", sd_0["fc2.weight"])
    print("fc2.weight at t=1:", sd_1["fc2.weight"])

    # Demonstrate training so the network can learn a time-varying mapping
    optimizer = optim.SGD(time_net.parameters(), lr=1e-1)
    for name, param in time_net.time_params.items():
        print(f"{name} : {param}")

    # Toy target y(t) = t * sum(x)
    for step in range(100):
        optimizer.zero_grad()

        x_train = torch.randn(10, 2)
        t_train = torch.rand(1).item()  # scalar in [0, 1)
        target = t_train * x_train.sum(dim=1, keepdim=True)

        y_pred = time_net(t_train, x_train)
        loss = F.mse_loss(y_pred, target)
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"Step {step + 1:3d}, t={t_train:.2f}, Loss={loss.item():.4f}")

    print("\nTraining complete. The time-parameterized network can now "
          "adjust weights as a function of time.")
    for name, param in time_net.time_params.items():
        print(f"{name} : {param}")
