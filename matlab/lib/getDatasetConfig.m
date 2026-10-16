function cfg = getDatasetConfig(dataset)
% getDatasetConfig  Single source of truth for per-dataset experiment settings.
%
% CDR and DCR follow Newman et al. 2020 (examples/GNvpro/EVP_{CDR,DCR}_GNvpro.m
% in Meganet.m) so that our numbers are comparable to the published baseline.
% ELM is introduced by this paper.
%
% Deviation from Newman et al., common to all datasets: nt = 12 time steps
% (Newman uses 2 plus 3 prolongation levels), because weight parameterization
% in time needs nt >> degree.
%
% alpha1 = alpha2 = 1e-10 as in Newman et al., selected on validation error over
% {1e-10, 1e-8, 1e-6, 1e-5, 1e-4} (see runPaperSweep.m and the appendix on
% selected settings).

switch upper(dataset)
    case 'ELM'
        cfg = struct('nTrain',[],'nVal',[],'nc',15);   % [] -> setupNNERDS's fixed split
    case 'CDR'
        cfg = struct('nTrain',400,'nVal',200,'nc',8);  % Newman et al.
    case 'DCR'
        cfg = struct('nTrain',8000,'nVal',1000,'nc',16); % Newman et al.
    otherwise
        error('getDatasetConfig:unknownDataset','Unknown dataset: %s',dataset);
end

% ADAM epoch budget, fixed per dataset and identical across every basis,
% architecture, optimizer and seed within that dataset. Early stopping is
% disabled; the budget is the only stopping rule, and the reported model is the
% best-validation iterate. Newman et al.'s budgets (maxWorkUnits, miniBatch=2,
% nt=2, 3-level prolongation) do not transfer to miniBatch=32, nt=12, one level.
switch upper(dataset)
    case 'DCR', cfg.maxEpochs = 2000;
    otherwise,  cfg.maxEpochs = 10000;
end
cfg.dataset = upper(dataset);
cfg.nt      = 12;
cfg.alpha1  = 1e-10;   % Newman et al.; validated, see above
cfg.alpha2  = 1e-10;   % Newman et al.

% Whether the ADAM arm initializes the readout W with the regularized
% least-squares solution, selected per dataset on validation error (see
% runPaperSweep.m and the appendix on selected settings). GNvpro is unaffected
% because variable projection eliminates the readout exactly at every iteration.
switch upper(dataset)
    case 'CDR', cfg.warmStartW = false;
    otherwise,  cfg.warmStartW = true;
end
