function runPrecisionMonomialSweep(workerIdx,nWorkers,sweepId)
% runPrecisionMonomialSweep  Single against double precision for the monomial basis.
%
% Block D of runPaperSweep runs CDR in single precision for the Legendre basis,
% where cond(V_L) <= 67.5 and the rounding estimate cond(V)*eps_single stays below
% 1e-5. On the normalized interval the monomial basis has the largest condition
% number (nt = 12):
%
%   d                    10        11
%   cond(V_M)          8352     40755
%   cond*eps_single  1.0e-3    4.9e-3
%
% Both precisions run in this sweep, so each single-precision run has a
% double-precision partner with the same seed, code and MATLAB version.
% The paper sweep stops the monomial basis at d=8.
%
% CDR, T = 1, both architectures, both optimizers, seeds 20:29, budgets and alpha
% from getDatasetConfig. 2 x 2 x 2 x 2 x 10 = 160 runs.
if nargin<3 || isempty(sweepId), sweepId = 'precision-monomial'; end
addpath(genpath('..'));
addpath(fullfile('..','..','data'));

seeds = 20:29; T = 1; ds = 'CDR'; basis = 'monomial';
ARCH = {'ResNN','hamiltonian'};
OPTI = {'sgd','GNvpro'};
PREC = {[],'single'};

todo = {};
for d = 10:11
    for j = 1:numel(ARCH)
        for k = 1:numel(OPTI)
            for q = 1:numel(PREC)
                for s = seeds
                    todo{end+1} = {ARCH{j},OPTI{k},d,PREC{q},s}; %#ok<AGROW>
                end
            end
        end
    end
end

p = sweepPaths(sweepId);
mine = workerIdx:nWorkers:numel(todo);
fprintf('worker %d/%d: %d of %d run(s)\n',workerIdx,nWorkers,numel(mine),numel(todo));

nOK = 0; nFail = 0;
for i = mine
    [dyn,opti,d,prec,seed] = deal(todo{i}{:});
    % Mirror runExperiment's own tagging so the skip check finds its file.
    tag = runTag(ds,dyn,basis,d,T,opti,seed);
    if ~isempty(prec) && strcmpi(prec,'single'), tag = [tag '_fp32']; end
    if exist(fullfile(p.raw,[tag '.mat']),'file'), nOK = nOK + 1; continue, end
    fprintf('  %s\n',tag);
    try
        out = evalc('runExperiment(dyn,T,d,opti,basis,ds,seed,sweepId,[],prec);');
        fid = fopen(fullfile(p.logs,[tag '.log']),'w');
        if fid~=-1, fprintf(fid,'%s\n%s',provenanceLine('%'),out); fclose(fid); end
        nOK = nOK + 1;
    catch ME
        fprintf(2,'  *** FAILED %s: %s\n',tag,ME.message);
        nFail = nFail + 1;
    end
end
fprintf('WORKER %d DONE ok=%d fail=%d\n',workerIdx,nOK,nFail);
