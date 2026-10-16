function p = sweepPaths(sweepId)
% sweepPaths  Canonical artifact locations for one sweep.
%
%   results/runs.csv                 ledger, every run of every sweep
%   results/sweeps/<id>.json         manifest of one sweep
%   results/raw/<id>/matlab/*.mat    trained weights (gitignored)
%   results/raw/<id>/logs/*.log      stdout (gitignored)
%
% Runs from different sweeps must not be mixed into one table, because they may
% come from different code versions.
root      = fullfile(fileparts(mfilename('fullpath')),'..','..');
p.root    = root;
p.ledger  = fullfile(root,'results','runs.csv');
p.sweepId = sweepId;
p.manifest= fullfile(root,'results','sweeps',[sweepId '.json']);
p.raw     = fullfile(root,'results','raw',sweepId,'matlab');
p.logs    = fullfile(root,'results','raw',sweepId,'logs');
for f = {'raw','logs'}
    if ~exist(p.(f{1}),'dir'), mkdir(p.(f{1})); end
end
