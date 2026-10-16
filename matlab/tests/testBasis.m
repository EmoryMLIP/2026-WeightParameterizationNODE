function testBasis()
% Unit tests for getPolynomialBasis + the Theta = theta*A parameterization
% + the Gram-matrix Tikhonov regularizer.  Run: testBasis
addpath(genpath('..'));
np = 0; nf = 0;
function chk(name,cond,detail)
    if cond, np=np+1; fprintf('  PASS  %s\n',name);
    else,    nf=nf+1; fprintf(2,'  FAIL  %s   %s\n',name,detail); end
end

fprintf('\n== 1. A is T-independent and correctly sized ==\n');
for basis = {'monomial','Legendre'}
    b = basis{1};
    A1 = getPolynomialBasis(3,1,12,b);
    A5 = getPolynomialBasis(3,5,12,b);
    A10= getPolynomialBasis(3,10,12,b);
    chk(sprintf('%s: size = nt x (d+1)',b), isequal(size(A1),[12 4]), mat2str(size(A1)));
    chk(sprintf('%s: A independent of T',b), norm(A1-A5,'fro')+norm(A1-A10,'fro') < 1e-12, ...
        sprintf('%.2e',norm(A1-A10,'fro')));
    chk(sprintf('%s: cond(A) = %.2f, T-independent',b,cond(A1)), ...
        abs(cond(A1)-cond(A10)) < 1e-8, sprintf('%.3f vs %.3f',cond(A1),cond(A10)));
end

fprintf('\n== 2. Both bases span the same space (degree-3 polynomials) ==\n');
Am = getPolynomialBasis(3,7,12,'monomial');
Al = getPolynomialBasis(3,7,12,'Legendre');
chk('range(A_mono) == range(A_leg)', ...
    norm(Am - Al*(Al\Am),'fro')/norm(Am,'fro') < 1e-12, ...
    sprintf('%.2e',norm(Am - Al*(Al\Am),'fro')/norm(Am,'fro')));

fprintf('\n== 3. Gram matrix matches high-accuracy quadrature ==\n');
for basis = {'monomial','Legendre'}
    b = basis{1}; T = 3.7; d = 4;
    [~,M] = getPolynomialBasis(d,T,12,b);
    pf = @(j,tt) basisFun(b,j,tt,T);
    Mq = zeros(d+1);
    for a=0:d
        for c=0:d
            Mq(a+1,c+1) = integral(@(tt) pf(a,tt).*pf(c,tt),0,T,'AbsTol',1e-14,'RelTol',1e-12);
        end
    end
    chk(sprintf('%s: ||M - M_quad|| / ||M||',b), norm(M-Mq,'fro')/norm(Mq,'fro') < 1e-10, ...
        sprintf('%.2e',norm(M-Mq,'fro')/norm(Mq,'fro')));
end
[~,ML] = getPolynomialBasis(3,2.0,12,'Legendre');
chk('Legendre Gram is diagonal', norm(ML-diag(diag(ML)),'fro') < 1e-14, 'off-diagonal mass');
[~,MM] = getPolynomialBasis(3,2.0,12,'monomial');
chk('monomial Gram is dense (not diagonal)', norm(MM-diag(diag(MM)),'fro') > 1e-3, 'unexpectedly diagonal');
fprintf('     cond(M_leg) = %.3f   cond(M_mono) = %.3f\n', cond(ML), cond(MM));

fprintf('\n== 3b. the unnormalized (naive) monomial arm ==\n');
cr = arrayfun(@(T) cond(getPolynomialBasis(3,T,12,'monomial-raw')), [1 5 10]);
cn = arrayfun(@(T) cond(getPolynomialBasis(3,T,12,'monomial')),     [1 5 10]);
chk('monomial-raw: cond grows with T (the units artifact)', cr(3) > 10*cr(1), ...
    sprintf('%.1f -> %.1f',cr(1),cr(3)));
chk('monomial: cond independent of T', max(cn)-min(cn) < 1e-9, sprintf('%.3f..%.3f',min(cn),max(cn)));
Ar = getPolynomialBasis(3,10,12,'monomial-raw'); An = getPolynomialBasis(3,10,12,'monomial');
chk('raw and normalized span the SAME space', ...
    norm(An-Ar*(Ar\An),'fro')/norm(An,'fro') < 1e-12, 'spans differ');
T4 = 3.7; d4 = 4; [~,Mr] = getPolynomialBasis(d4,T4,12,'monomial-raw');
Mq = zeros(d4+1);
for a=0:d4, for b=0:d4, Mq(a+1,b+1) = integral(@(x) x.^a.*x.^b,0,T4,'AbsTol',1e-14); end, end
chk('monomial-raw Gram matches quadrature', norm(Mr-Mq,'fro')/norm(Mq,'fro') < 1e-10, ...
    sprintf('%.2e',norm(Mr-Mq,'fro')/norm(Mq,'fro')));

fprintf('\n== 4. Theta = theta*A reproduces the intended weight trajectory ==\n');
nc = 5; nt = 12; d = 3; T = 4;
A = getPolynomialBasis(d,T,nt,'Legendre');
K = dense([nc,nc]); layer = singleLayer(K,'Bin',eye(nc));
blk = ResNN(layer,nt,T/nt,'A',A');
nL  = nTheta(layer);
chk('nTheta = (d+1)*nTheta(layer)', nTheta(blk) == (d+1)*nL, ...
    sprintf('%d vs %d',nTheta(blk),(d+1)*nL));
Th  = randn(nL,d+1);
Traj = reshape(Th(:),nL,d+1) * A';        % what ResNN forms internally
tj  = linspace(0,T,nt);
tauj= 2*tj/T - 1;
Ref = zeros(nL,nt);
for j=1:nt
    for e=0:d, Ref(:,j) = Ref(:,j) + Th(:,e+1)*legendreP(e,tauj(j)); end
end
chk('theta*A'' == sum_d theta_d p_d(t_j)', norm(Traj-Ref,'fro')/norm(Ref,'fro') < 1e-12, ...
    sprintf('%.2e',norm(Traj-Ref,'fro')/norm(Ref,'fro')));

fprintf('\n== 5. initTheta reproduces a constant-in-time trajectory exactly ==\n');
for basis = {'monomial','Legendre'}
    b = basis{1};
    for T = [1 10]
        Ab = getPolynomialBasis(d,T,nt,b);
        blkb = ResNN(layer,nt,T/nt,'A',Ab');
        rng(0); th0 = initTheta(blkb);
        Traj0 = reshape(th0,nL,d+1)*Ab';
        spread = max(max(abs(Traj0 - mean(Traj0,2))));
        chk(sprintf('%s T=%d: init trajectory constant in t',b,T), spread < 1e-10, ...
            sprintf('max deviation %.2e',spread));
    end
end

fprintf('\n== 6. tikhonovReg with B = kron(chol(M),I) gives Hessian alpha*(M kron I) ==\n');
alpha = 0.3; nth = 6; d = 3;
[~,M] = getPolynomialBasis(d,2.5,12,'monomial');
M = (M+M')/2;
R = chol(M);                                  % R'*R = M
B = kron(R,speye(nth));
reg = tikhonovReg(B,alpha);
Hwant = alpha*kron(M,eye(nth));
Hgot  = full(getA(reg));
chk('Hessian == alpha*kron(M,I)', norm(Hgot-Hwant,'fro')/norm(Hwant,'fro') < 1e-12, ...
    sprintf('%.2e',norm(Hgot-Hwant,'fro')/norm(Hwant,'fro')));
Th = randn(nth,d+1); x = Th(:);
[Sc,~,dS] = regularizer(reg,x);
% closed form: 0.5*alpha*sum_{d,e} M_de theta_d'theta_e
Swant = 0.5*alpha*trace(Th*M*Th');
chk('R(x) == 0.5*alpha*sum M_de theta_d''theta_e', abs(Sc-Swant)/abs(Swant) < 1e-12, ...
    sprintf('%.6g vs %.6g',Sc,Swant));
% gradient check
g = 1e-6; v = randn(size(x));
[S1] = regularizer(reg,x+g*v); [S0] = regularizer(reg,x-g*v);
chk('gradient consistent with finite differences', ...
    abs((S1-S0)/(2*g) - dS'*v)/abs(dS'*v) < 1e-6, ...
    sprintf('%.3e vs %.3e',(S1-S0)/(2*g),dS'*v));
chk('passing M instead of chol(M) would be WRONG (M''M ~= M)', ...
    norm(M'*M - M,'fro')/norm(M,'fro') > 1e-3, 'M happens to be idempotent');

fprintf('\n== 7. baseline (none) Gram = quadrature weight of the same functional ==\n');
[An,Mn] = getPolynomialBasis([],6,12,'none');
chk('none: A = eye(nt)', isequal(An,eye(12)), 'not identity');
chk('none: M = (T/nt)*eye(nt)', norm(Mn - 0.5*eye(12),'fro') < 1e-14, mat2str(Mn(1,1)));

fprintf('\n---------------------------------------------\n');
fprintf('%d passed, %d failed\n', np, nf);
if nf>0, error('testBasis:failures','%d test(s) failed',nf); end

function y = basisFun(b,j,tt,T)
    tau = 2*tt/T - 1;
    if strcmpi(b,'monomial'), y = tau.^j; else, y = legendreP(j,tau); end
end
end