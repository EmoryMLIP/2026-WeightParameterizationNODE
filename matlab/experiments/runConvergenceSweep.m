function sweepId = runConvergenceSweep(workerIdx,nWorkers,sweepId)
% runConvergenceSweep  Training at five times the ADAM budget, to check convergence.
%
% Every arm trains to a fixed epoch budget with early stopping disabled. If the
% best-validation iterate falls at the end of the budget, a ranking of arms
% reflects descent speed. This sweep retrains the endpoints of the degree axis
% (D=0, D=11) and the unconstrained model at five times the budget on CDR and DCR.
%
% ADAM only: GNvpro is bounded by maxWorkUnits, not epochs, so raising maxEpochs
% would not change its runs.
if nargin<3 || isempty(sweepId), sweepId = 'convergence'; end
addpath(genpath('..'));
addpath(fullfile('..','..','data'));

seeds = [20 21 22 23 24];  T = 1;  opti = 'sgd';
% {dataset, budget}: five times the budget in getDatasetConfig.
sets = { {'DCR',10000}, {'CDR',50000} };
% {basis, degree}: the static endpoint, the equivalent endpoint, and the
% unconstrained arm they are measured against.
arms = { {'Legendre',0}, {'Legendre',11}, {'none',0} };

todo = {};
for k = 1:numel(sets)
    for dyn = {'ResNN','hamiltonian'}
        for a = 1:numel(arms)
            for s = seeds
                todo{end+1} = {sets{k}{1},dyn{1},arms{a}{1},arms{a}{2},sets{k}{2},s}; %#ok<AGROW>
            end
        end
    end
end
p = sweepPaths(sweepId);
mine = workerIdx:nWorkers:numel(todo);
fprintf('worker %d/%d: %d of %d run(s)\n',workerIdx,nWorkers,numel(mine),numel(todo));

nOK = 0; nFail = 0;
for i = mine
    c = todo{i};
    % Mirror runExperiment's budget tag (_e<maxEpochs>) so the skip check finds its file.
    tag = sprintf('%s_e%d',runTag(c{1},c{2},c{3},c{4},T,opti,c{6}),c{5});
    if exist(fullfile(p.raw,[tag '.mat']),'file'), nOK = nOK + 1; continue, end
    fprintf('  %s\n',tag);
    try
        out = evalc('runExperiment(c{2},T,c{4},opti,c{3},c{1},c{6},sweepId,c{5});');
        fid = fopen(fullfile(p.logs,[tag '.log']),'w');
        if fid~=-1, fprintf(fid,'%s\n%s',provenanceLine('%'),out); fclose(fid); end
        nOK = nOK + 1;
    catch ME
        fprintf(2,'  *** FAILED %s: %s\n',tag,ME.message);
        nFail = nFail + 1;
    end
end
fprintf('WORKER %d DONE ok=%d fail=%d\n',workerIdx,nOK,nFail);
