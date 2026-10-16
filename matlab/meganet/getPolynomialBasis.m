function [A,M] = getPolynomialBasis(d,T,nt,basis)
% getPolynomialBasis  Time-dependent weight parameterization basis.
%
%   [A,M] = getPolynomialBasis(d,T,nt,basis)
%
%   A : nt x (d+1) basis matrix, A(j,k) = p_{k-1}(t_j), t_j = linspace(0,T,nt)
%   M : (d+1) x (d+1) Gram matrix, M(d,e) = \int_0^T p_d(t) p_e(t) dt
%
%   Both bases are evaluated on the SAME normalized coordinate
%       tau(t) = 2*t/T - 1  in  [-1,1],
%   so that A (and hence cond(A)) does not depend on T, and monomial vs
%   Legendre is a comparison of basis functions rather than of domains.
%
%   M is exact in closed form (affine change of variables dt = (T/2) dtau):
%       Legendre : M = T/(2d+1) * delta_de                     (diagonal)
%       monomial : M = T/(d+e+1) if d+e even, 0 otherwise      (dense)
%
%   basis = 'monomial-raw' is the NAIVE, unnormalized monomial basis: p_d(t) = t^d
%   evaluated on the raw interval [0,T], as the construction is conventionally
%   written. It is included as an explicit experimental arm, not as a mistake to
%   avoid: V[0,T] = V[0,1]*diag(1,T,...,T^D), so cond grows like T^D purely
%   through a choice of units. Comparing it against 'monomial' isolates the
%   effect of time normalization from the effect of orthogonality.
%
%   basis = 'none' returns A = eye(nt) (non-parameterized baseline); the
%   free parameters are then the weights at the nt time points themselves,
%   and M = (T/nt)*eye(nt) is the corresponding midpoint quadrature of the
%   same continuous functional \int_0^T ||theta(t)||^2 dt.

if not(exist('basis','var')) || isempty(basis), basis = 'monomial'; end

if strcmpi(basis,'none')
    A = eye(nt);
    M = (T/nt)*eye(nt);
    return
end

t   = linspace(0,T,nt);
tau = 2*t/T - 1;               % [0,T] -> [-1,1]
k   = 0:d;

switch lower(basis)
    case 'monomial'
        A = tau(:).^k;
        [D,E] = ndgrid(k,k);
        S     = D + E;
        M     = (T./(S+1)) .* (mod(S,2)==0);
    case 'monomial-raw'
        % Deliberately NOT normalized: powers of raw t on [0,T].
        A = t(:).^k;
        % Gram matrix on the raw interval: M_de = int_0^T t^(d+e) dt.
        [D,E] = ndgrid(k,k);
        S     = D + E;
        M     = T.^(S+1) ./ (S+1);
    case 'legendre'
        A = zeros(nt,d+1);
        for j = k
            A(:,j+1) = reshape(legendreP(j,tau),[],1);
        end
        M = diag(T./(2*k+1));
    otherwise
        error('getPolynomialBasis:invalidBasis','invalid basis "%s"',basis);
end
