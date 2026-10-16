function sweepId = runGnvproSeeds(workerIdx,nWorkers,sweepId)
% runGnvproSeeds  GNvpro seeds 30:59 on the T=1 degree axis.
%
% Continues seeds 20:29 of runPaperSweep. Everything else is the configured
% default: double precision, alpha = 1e-10, the dataset's warm start.
if nargin<3 || isempty(sweepId), sweepId = 'gnvpro-seeds'; end
addpath(genpath('..'));
addpath(fullfile('..','..','data'));

seeds = 30:59;
DS   = {'ELM','CDR','DCR'};
ARCH = {'ResNN','hamiltonian'};

todo = {};
for i = 1:numel(DS)
    for j = 1:numel(ARCH)
        for s = seeds
            for d = 0:11
                todo{end+1} = {DS{i},ARCH{j},'Legendre',d,s}; %#ok<AGROW>
            end
            todo{end+1} = {DS{i},ARCH{j},'none',0,s}; %#ok<AGROW>
        end
    end
end

p = sweepPaths(sweepId);
mine = workerIdx:nWorkers:numel(todo);
fprintf('worker %d/%d: %d of %d run(s)\n',workerIdx,nWorkers,numel(mine),numel(todo));

nOK = 0; nFail = 0;
for i = mine
    [ds,dyn,basis,d,seed] = deal(todo{i}{:});
    tag = runTag(ds,dyn,basis,d,1,'GNvpro',seed);
    if exist(fullfile(p.raw,[tag '.mat']),'file'), nOK = nOK + 1; continue, end
    fprintf('  %s\n',tag);
    try
        out = evalc('runExperiment(dyn,1,d,''GNvpro'',basis,ds,seed,sweepId);');
        fid = fopen(fullfile(p.logs,[tag '.log']),'w');
        if fid~=-1, fprintf(fid,'%s\n%s',provenanceLine('%'),out); fclose(fid); end
        nOK = nOK + 1;
    catch ME
        fprintf(2,'  *** FAILED %s: %s\n',tag,ME.message);
        nFail = nFail + 1;
    end
end
fprintf('WORKER %d DONE ok=%d fail=%d\n',workerIdx,nOK,nFail);
