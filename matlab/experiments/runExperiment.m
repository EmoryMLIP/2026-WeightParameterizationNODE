function results = runExperiment(dynamic,T,d,opti,basis,dataset,seed,sweepId,maxEpochs,precision,alphaOverride,warmStartW)
% runExperiment  Train a discretize-then-optimize ResNet / Hamiltonian network
%                on a surrogate task and save the run record.
%
%   dynamic : 'ResNN' | 'hamiltonian'
%   T       : final time (step size is T/nt)
%   d       : polynomial degree (ignored when basis = 'none')
%   opti    : 'sgd' (ADAM) | 'GNvpro'
%   basis   : 'monomial' | 'Legendre' | 'none'
%   dataset : 'ELM' | 'CDR' | 'DCR'
%   seed    : network-init / minibatch seed (default 20). The data split is
%             always rng(42) and does not depend on this.
%   sweepId : optional; which sweep this run belongs to (see newSweep). Results
%             land in results/raw/<sweepId>/. Defaults to 'scratch', which is
%             gitignored and never enters the ledger.
%   maxEpochs, precision ('double' | 'single'), alphaOverride ([alpha1 alpha2]),
%   warmStartW : optional; default to the values in getDatasetConfig.
%
% For both optimizers the reported iterate has the lowest validation objective.

maxNumCompThreads(1);   % ADAM is not reproducible under multithreaded BLAS

if nargin < 5 || isempty(basis),   basis   = 'monomial'; end
if nargin < 6 || isempty(dataset), dataset = 'ELM';      end
if nargin < 7 || isempty(seed),    seed    = 20;         end
if nargin < 8 || isempty(sweepId), sweepId = 'scratch';  end

paths = sweepPaths(sweepId);

cfg = getDatasetConfig(dataset);
nt = cfg.nt; nc = cfg.nc; alpha1 = cfg.alpha1; alpha2 = cfg.alpha2;
if nargin < 9  || isempty(maxEpochs), maxEpochs = cfg.maxEpochs; end
if nargin < 10 || isempty(precision), precision = 'double'; end
if nargin < 12 || isempty(warmStartW), warmStartW = cfg.warmStartW; end
% alphaOverride = [alpha1 alpha2]; used by the regularization-selection sweep.
if nargin >= 11 && ~isempty(alphaOverride)
    alpha1 = alphaOverride(1); alpha2 = alphaOverride(2);
end

%% ---- data (split fixed by rng(42), independent of the training seed) ----
rng(42);
switch cfg.dataset
    case 'ELM', [Yt,Ct,Yv,Cv,Ytest,Ctest] = setupNNERDS();
    case 'CDR', [Yt,Ct,Yv,Cv,Ytest,Ctest] = setupCDR(cfg.nTrain,cfg.nVal);
    case 'DCR', [Yt,Ct,Yv,Cv,Ytest,Ctest] = setupDCR(cfg.nTrain,cfg.nVal);
end

seedUsed = seed;
rng(seedUsed);

%% ---- network ----
block1 = NN({singleLayer(dense([nc,size(Yt,1)],'Bin',eye(nc)))});
if strcmpi(basis,'none'), d = 0; end
[A,M] = getPolynomialBasis(d,T,nt,basis);

switch lower(dynamic)
    case 'resnn'
        block2 = ResNN(singleLayer(dense([nc,nc]),'Bin',eye(nc)),nt,T/nt,'A',A');
    case 'hamiltonian'
        block2 = HamiltonianNN(@tanhActivation,dense([nc,nc]),eye(nc),nt,T/nt,'A',A');
    otherwise
        error('runExperiment:unknownDynamic','Unknown dynamic: %s',dynamic);
end
net   = Meganet({block1,block2});
pLoss = regressionLoss();
% Single precision is where basis conditioning can matter: rounding error scales
% as cond(V) times machine epsilon.
if strcmpi(precision,'single')
    net.precision = 'single';
    [Yt,Ct,Yv,Cv,Ytest,Ctest] = deal(single(Yt),single(Ct),single(Yv), ...
                                     single(Cv),single(Ytest),single(Ctest));
end

%% ---- regularizer: manuscript Eq. (5) ----
%   R(theta) = alpha1/2 * ||theta_static||^2
%            + alpha1/2 * int_0^T ||theta_NODE(t)||^2 dt
% The integral term expands via the Gram matrix M (Eq. 18) as
%   sum_{d,e} M_de <theta_d, theta_e> = vec(Theta)' (M kron I) vec(Theta).
% tikhonovReg(B,alpha) implements alpha/2*||B x||^2 with Hessian alpha*(B'B),
% so B must be the square root (Cholesky factor) of the Gram matrix, not M.
n1     = nTheta(block1);
nLayer = nTheta(block2) / size(A,2);
B      = blkdiag(speye(n1), kron(sparse(chol(M)), speye(nLayer)));
% dnnObjFctn requires net, pRegTheta and pRegW to share one precision, so the
% regularizer operators are cast with the network.
if strcmpi(precision,'single'), B = single(full(B)); end
pRegTheta = tikhonovReg(B,alpha1);

th0 = initTheta(net);

startTime = tic;
switch lower(opti)
    case 'sgd'
        % Warm start: the regularized least-squares solve that GNvpro's
        % variable-projection step performs, so both optimizers start from the
        % same readout.
        YN0 = forwardProp(net,th0,Yt);
        X0  = [YN0; ones(1,size(YN0,2))];
        if warmStartW
            W = (Ct*X0') / (X0*X0' + alpha2*eye(size(X0,1)));
        else
            % Ablation arm: PyTorch's nn.Linear initialization, U(-b,b) with
            % b = 1/sqrt(fan_in), to match the PyTorch pipeline.
            b = 1/sqrt(size(X0,1)-1);
            W = (2*b)*rand(size(Ct,1),size(X0,1)) - b;
        end
        if strcmpi(precision,'single'), W = single(W); end
        pRegW = tikhonovReg(opEye(numel(W)),alpha2);

        opt = sgd('nesterov',false,'ADAM',true,'miniBatch',32,'out',1,'lossTol',-Inf);
        opt.learningRate      = 0.001;
        % Fixed budget for every arm with early stopping disabled. The reported
        % model is the best-validation iterate (infoOptLoss).
        opt.maxEpochs         = maxEpochs;
        opt.earlyStopPatience = Inf;

        fctn = dnnObjFctn(net,pRegTheta,pLoss,pRegW,Yt,Ct);
        fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);
        [~,his,~,infoOptLoss] = solve(opt,fctn,[th0;W(:)],fval);
        [thOpt,WOpt] = split(fctn,infoOptLoss.xOptLoss);
        bestEpoch    = infoOptLoss.optEpoch;

    case 'gnvpro'
        opt = trnewton();
        opt.linSol       = GMRES('m',20,'tol',1e-2);
        opt.out          = 1;
        opt.maxIter      = Inf;
        opt.maxWorkUnits = 400;    % Newman et al.
        opt.atol         = 1e-16;
        opt.rtol         = 1e-16;

        % Btheta is the Gram-matrix square root used for ADAM, so both optimizers
        % regularize the same functional.
        fctn = dnnVarProRegressionObjFctn(net,pLoss,Yt,Ct, ...
                   'alpha1',alpha1,'alpha2',alpha2,'Btheta',B);
        fval = dnnObjFctn(net,[],pLoss,[],Yv,Cv);
        [~,his,~,~,infoOptLoss] = solve(opt,fctn,th0,fval);
        thOpt      = infoOptLoss.xOptLoss;
        bestEpoch  = infoOptLoss.optEpoch;
        [~,para]   = eval(fctn,thOpt);
        WOpt       = reshape(para.W,size(Ct,1),[]);

    otherwise
        error('runExperiment:unknownOptimizer','Unknown optimizer: %s',opti);
end
elapsedSeconds = toc(startTime);

%% ---- metrics ----
pred = @(Y) reshape(WOpt,size(Ct,1),[]) * [forwardProp(net,thOpt,Y); ones(1,size(Y,2))];
err.train = relativeErrors(pred(Yt),Ct);
err.val   = relativeErrors(pred(Yv),Cv);
err.test  = relativeErrors(pred(Ytest),Ctest);
triv.train = trivialPredictorError(Ct,Ct);
triv.val   = trivialPredictorError(Ct,Cv);
triv.test  = trivialPredictorError(Ct,Ctest);

fprintf('\n%-8s%-12s%-12s%-12s   (trivial: %.4f)\n','','aggregate','median','meanRatio',triv.test.aggregate);
for f = {'train','val','test'}
    e = err.(f{1});
    fprintf('%-8s%-12.4f%-12.4f%-12.4f\n',f{1},e.aggregate,e.median,e.meanRatio);
end

%% ---- assemble and save ----
config = struct('dataset',cfg.dataset,'dynamic',dynamic,'T',T,'d',d,'opti',opti, ...
                'basis',basis,'nt',nt,'nc',nc,'alpha1',alpha1,'alpha2',alpha2, ...
                'nTrain',size(Yt,2),'nVal',size(Yv,2),'nTest',size(Ytest,2), ...
                'seed',seedUsed,'maxEpochs',maxEpochs,'precision',precision,'bestEpoch',bestEpoch,'elapsedSeconds',elapsedSeconds, ...
                'warmStartW',warmStartW, ...
                'sweepId',sweepId,'commit',getfield(provenance(),'commit'));

results = struct('config',config,'net',net,'thOpt',thOpt,'WOpt',WOpt,'his',his, ...
                 'err',err,'trivial',triv, ...
                 'nParamsDynamics',nTheta(block2), ...      % theta_NODE only
                 'nParamsTheta',numel(thOpt), ...           % theta_NODE + block1
                 'nParamsTotal',numel(thOpt)+numel(WOpt));  % + readout

tag = runTag(cfg.dataset,dynamic,basis,d,T,opti,seedUsed);
if strcmpi(precision,'single'), tag = [tag '_fp32']; end
% Tags mark only deviations from the configured default, so runs at a
% non-default setting do not overwrite the default run.
if warmStartW ~= cfg.warmStartW, tag = sprintf('%s_ws%d',tag,warmStartW); end
if maxEpochs ~= cfg.maxEpochs, tag = sprintf('%s_e%d',tag,maxEpochs); end
% Workers running the same configuration at different alpha would otherwise
% write to one file.
if nargin >= 11 && ~isempty(alphaOverride)
    tag = sprintf('%s_a%g',tag,alpha1);
end
save(fullfile(paths.raw,[tag '.mat']),'results');
fprintf('Saved results/raw/%s/matlab/%s.mat\n',sweepId,tag);
