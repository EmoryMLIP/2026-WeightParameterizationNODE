function tbl = collectResults(rawRoot)
% collectResults  Walk results/raw/*/matlab/*.mat into the ledger results/runs.csv.
%
% One row per run, across every sweep. Error columns follow relativeErrors.m:
%   *_agg   aggregate ||Pred-C||_F/||C||_F
%   *_med   median per-sample relative error
%   *_mean  mean per-sample relative error
% trivial_* is the same metric for the "predict the training mean" reference.
%
% nParamsDynamics is theta_NODE only; nParamsTheta adds the input block;
% nParamsTotal adds the readout. Quote one and say which.

root = fullfile(fileparts(mfilename('fullpath')),'..','..');
if nargin<1 || isempty(rawRoot), rawRoot = fullfile(root,'results','raw'); end
sweeps = dir(rawRoot);
% Directories prefixed 'scratch' hold one-off checks and stay out of the ledger.
sweeps = sweeps([sweeps.isdir] & ~ismember({sweeps.name},{'.','..'}) ...
                & ~startsWith({sweeps.name},'scratch'));

rows = {};  nSkipped = 0;
for s = 1:numel(sweeps)
    files = dir(fullfile(rawRoot,sweeps(s).name,'matlab','*.mat'));
    drop  = supersededPatterns(root,sweeps(s).name);
    for i = 1:numel(files)
        % Records a later sweep replaced stay on disk but must not reach the
        % ledger, where two versions of one cell make the seed count ambiguous.
        if ~isempty(drop) && any(cellfun(@(pat) ~isempty(regexp(files(i).name,pat,'once')), drop))
            nSkipped = nSkipped + 1; continue
        end
        L = load(fullfile(files(i).folder,files(i).name));
        if ~isfield(L,'results') || ~isfield(L.results,'err'), continue, end
        r = L.results; c = r.config;
        rows{end+1,1} = { sweeps(s).name, getfielddef(c,'commit','unknown'), ...
            c.dataset, c.dynamic, c.basis, c.d, c.T, c.opti, c.seed, ...
            getfielddef(c,'precision','double'), ...
            c.nTrain, c.nVal, c.nTest, c.nc, c.nt, c.alpha1, c.alpha2, ...
            getfielddef(c,'maxEpochs',NaN), ...
            ... % warmStartW separates warm-start runs from default runs.
            double(getfielddef(c,'warmStartW',NaN)), ...
            c.bestEpoch, c.elapsedSeconds, ...
            ... % double(): cell2table downcasts a whole column to single if any row
            ... % is single (single-precision runs), rounding the ledger to 7 digits.
            double(r.err.train.aggregate), double(r.err.val.aggregate), double(r.err.test.aggregate), ...
            double(r.err.train.median),    double(r.err.val.median),    double(r.err.test.median), ...
            double(r.err.train.meanRatio), double(r.err.val.meanRatio), double(r.err.test.meanRatio), ...
            double(r.trivial.test.aggregate), double(r.trivial.test.median), ...
            r.nParamsDynamics, r.nParamsTheta, r.nParamsTotal }; %#ok<AGROW>
    end
end
if nSkipped > 0
    fprintf('collectResults: skipped %d superseded record(s); see the sweep manifests.\n',nSkipped);
end
if isempty(rows)
    error('collectResults:noUsableRuns', ...
        ['No runs in the current results format found under %s. Either the sweep ' ...
         'has not been run, or the .mat files predate the schema change -- rerun ' ...
         'with runNamedSweep.sh runPaperSweep.'], rawRoot);
end

names = {'sweep','commit','dataset','dynamic','basis','d','T','opti','seed','precision', ...
         'nTrain','nVal','nTest','nc','nt','alpha1','alpha2','maxEpochs','warmStartW', ...
         'bestEpoch','elapsedSeconds', ...
         'train_agg','val_agg','test_agg','train_med','val_med','test_med', ...
         'train_mean','val_mean','test_mean','trivial_test_agg','trivial_test_med', ...
         'nParamsDynamics','nParamsTheta','nParamsTotal'};
tbl = cell2table(vertcat(rows{:}),'VariableNames',names);
tbl = sortrows(tbl,{'sweep','dataset','basis','T','dynamic','opti','seed'});
out = fullfile(root,'results','runs.csv');

% Write the error columns with 17 significant digits. writetable's default gives
% about seven, which cannot distinguish an exact rerun from agreement to seven
% digits.
errCols = {'train_agg','val_agg','test_agg','train_med','val_med','test_med', ...
           'train_mean','val_mean','test_mean','trivial_test_agg','trivial_test_med', ...
           'elapsedSeconds'};
for k = 1:numel(errCols)
    c = errCols{k};
    if ismember(c,tbl.Properties.VariableNames) && isnumeric(tbl.(c))
        tbl.(c) = arrayfun(@(x) string(sprintf('%.17g',x)), tbl.(c));
    end
end
writetable(tbl,out);
fprintf('collectResults: %d run(s) from %d sweep(s) -> results/runs.csv\n', ...
        height(tbl), numel(unique(tbl.sweep)));
checkProvenance(tbl);
end

function v = getfielddef(s,f,d)
if isfield(s,f), v = s.(f); else, v = d; end
end

function pats = supersededPatterns(root,sweepId)
% Regexes for run files that a sweep's manifest marks superseded:
%
%   "superseded": { "supersededBy": "<sweepId>", "reason": "...",
%                   "patterns": ["^CDR_"] }
%
% Patterns rather than a tag list because a later sweep normally replaces a
% whole dataset or basis.
pats = {};
f = fullfile(root,'results','sweeps',[sweepId '.json']);
if ~exist(f,'file'), return, end
m = jsondecode(fileread(f));
if ~isfield(m,'superseded'), return, end
sup = m.superseded;
assert(isfield(sup,'patterns') && isfield(sup,'supersededBy') && isfield(sup,'reason'), ...
    'collectResults:badSuperseded', ...
    ['Sweep %s has a "superseded" block without patterns, supersededBy and reason. ' ...
     'Retiring runs from the ledger must say which, by what, and why.'], sweepId);
pats = cellstr(sup.patterns);
end
