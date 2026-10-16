function sweepId = runPaperSweep(workerIdx,nWorkers,sweepId)
% runPaperSweep  All discretize-then-optimize runs of the manuscript, seeds 20:29.
%
%   A  tables/tableC1.tex: Legendre and monomial at d=3, T = 1, 5, 10; Legendre d = 4, 5, 6
%      at T=1; the unconstrained baseline at all three horizons.
%   B  Monomial on the raw interval, d=3, T = 1, 5, 10.
%   C  Degrees not in A, at T=1. A and C give d = 0..11.
%   D  Legendre d = 3..11 at T=1 in single precision, CDR only.
%   E  Regularization alpha in {1e-8, 1e-6, 1e-5, 1e-4}; 1e-10 is in A.
%   F  Readout warm start opposite to the dataset default (ADAM only: GNvpro
%      eliminates the readout).
%   G  Monomial on the normalized interval, d = 4..8, CDR only.
%
% Epoch budgets come from getDatasetConfig (see runConvergenceSweep).
if nargin<3 || isempty(sweepId), sweepId = 'paper'; end
addpath(genpath('..'));
addpath(fullfile('..','..','data'));

seeds = 20:29;
DS   = {'ELM','CDR','DCR'};
ARCH = {'ResNN','hamiltonian'};
OPTI = {'sgd','GNvpro'};

% Each entry: {basis, degree, T, precision, alphaOverride, warmStartOverride}
% [] means "the configured default".
base = {};
for T = [1 5 10]
    base{end+1} = {'Legendre',3,T,[],[],[]};      %#ok<AGROW>  A
    base{end+1} = {'monomial',3,T,[],[],[]};      %#ok<AGROW>  A
    base{end+1} = {'none',    0,T,[],[],[]};      %#ok<AGROW>  A
    base{end+1} = {'monomial-raw',3,T,[],[],[]};  %#ok<AGROW>  B
end
for d = [4 5]
    base{end+1} = {'Legendre',d,1,[],[],[]};      %#ok<AGROW>  A
end
base{end+1} = {'Legendre',6,1,[],[],[]};          %          A
for d = [0 1 2 7 8 9 10 11]
    base{end+1} = {'Legendre',d,1,[],[],[]};      %#ok<AGROW>  C
end
for a = [1e-8 1e-6 1e-5 1e-4]
    base{end+1} = {'Legendre',3,1,[],[a a],[]};   %#ok<AGROW>  E
end

todo = {};
for i = 1:numel(DS)
    for j = 1:numel(ARCH)
        for k = 1:numel(OPTI)
            for b = 1:numel(base)
                for s = seeds
                    todo{end+1} = [{DS{i},ARCH{j},OPTI{k}},base{b},{s}]; %#ok<AGROW>
                end
            end
            % D: single precision, CDR only, across the degrees where the
            % conditioning of the Legendre basis varies.
            if strcmp(DS{i},'CDR')
                for d = 3:11
                    for s = seeds
                        todo{end+1} = [{DS{i},ARCH{j},OPTI{k}}, ...
                                       {'Legendre',d,1,'single',[],[]},{s}]; %#ok<AGROW>
                    end
                end
                % G: monomial on the normalized interval, so the basis comparison
                % extends over degrees 4 to 8, where cond(monomial)/cond(Legendre)
                % grows from 3.1 at d=3 to 123 at d=8.
                for d = 4:8
                    for s = seeds
                        todo{end+1} = [{DS{i},ARCH{j},OPTI{k}}, ...
                                       {'monomial',d,1,[],[],[]},{s}]; %#ok<AGROW>
                    end
                end
            end
        end
        % F: warm-start deviation, ADAM only.
        for s = seeds
            todo{end+1} = [{DS{i},ARCH{j},'sgd'}, ...
                           {'Legendre',3,1,[],[],'flip'},{s}]; %#ok<AGROW>
        end
    end
end

p = sweepPaths(sweepId);
mine = workerIdx:nWorkers:numel(todo);
fprintf('worker %d/%d: %d of %d run(s)\n',workerIdx,nWorkers,numel(mine),numel(todo));

nOK = 0; nFail = 0;
for i = mine
    c = todo{i};
    [ds,dyn,opti,basis,d,T,prec,alphaOv,wsOv,seed] = deal(c{:});
    cfg = getDatasetConfig(ds);
    ws = [];
    if ischar(wsOv) && strcmp(wsOv,'flip'), ws = ~cfg.warmStartW; end
    % Mirror runExperiment's own tagging so the skip check finds its file.
    tag = runTag(ds,dyn,basis,d,T,opti,seed);
    if ~isempty(prec) && strcmpi(prec,'single'), tag = [tag '_fp32']; end
    if ~isempty(ws), tag = sprintf('%s_ws%d',tag,ws); end
    if ~isempty(alphaOv), tag = sprintf('%s_a%g',tag,alphaOv(1)); end
    if exist(fullfile(p.raw,[tag '.mat']),'file'), nOK = nOK + 1; continue, end
    fprintf('  %s\n',tag);
    try
        out = evalc(['runExperiment(dyn,T,d,opti,basis,ds,seed,sweepId,[],' ...
                     'prec,alphaOv,ws);']);
        fid = fopen(fullfile(p.logs,[tag '.log']),'w');
        if fid~=-1, fprintf(fid,'%s\n%s',provenanceLine('%'),out); fclose(fid); end
        nOK = nOK + 1;
    catch ME
        fprintf(2,'  *** FAILED %s: %s\n',tag,ME.message);
        nFail = nFail + 1;
    end
end
fprintf('WORKER %d DONE ok=%d fail=%d\n',workerIdx,nOK,nFail);
