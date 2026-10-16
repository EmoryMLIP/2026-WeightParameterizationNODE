function [T,sweepId] = selectDefaultRuns(T,sweepId)
% selectDefaultRuns  Keep only the runs the manuscript compares.
%
% The MATLAB counterpart of python/sweep_defaults.py. One sweep holds the
% regularization sweep, the single-precision repetition and the readout
% warm-start ablation, each differing from the default runs in a single column.
% Without this filter a cell averages runs trained at different alpha.
%
%   T = selectDefaultRuns(T)            pins the paper sweep
%   T = selectDefaultRuns(T,sweepId)    pins the named sweep
%
% The ledger's last sweep is not the default: the GNvpro seed top-up sorts after
% the paper sweep and contains no monomial runs.

if nargin < 2 || isempty(sweepId)
    sweepId = '2026-10-02-paper';   % PAPER_SWEEP in python/sweep_defaults.py
end

if ismember('sweep',T.Properties.VariableNames)
    T = T(strcmp(T.sweep,sweepId),:);
end

% The ablation runs flip each dataset's selected warmStartW. NaN marks records
% without the column, which used the default.
if ismember('warmStartW',T.Properties.VariableNames)
    keep = true(height(T),1);
    for ds = unique(T.dataset)'
        cfg = getDatasetConfig(ds{1});
        m = strcmp(T.dataset,ds{1});
        keep(m) = isnan(T.warmStartW(m)) | T.warmStartW(m) == cfg.warmStartW;
    end
    T = T(keep,:);
end

T = T(T.alpha1 == 1e-10 & strcmp(T.precision,'double'),:);
end
