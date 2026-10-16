"""Per-dataset experiment settings, mirroring matlab/lib/getDatasetConfig.m.

Keep this file and getDatasetConfig.m in step: the MATLAB (discretize-then-
optimize) and PyTorch (optimize-then-discretize) halves of the paper are
compared directly, so they must train on the same splits with the same width
and evaluate on identically preprocessed targets.

CDR and DCR follow Newman et al. 2020; ELM is introduced by this paper.
"""

# Split seed is FIXED and independent of the run seed, matching MATLAB's
# rng(42) before setup{CDR,DCR}. The per-run seed controls network
# initialization and minibatch order only, so seed-to-seed spread measures
# optimization variability rather than which samples landed in the test set.
SPLIT_SEED = 42

# warm_start / lr are the neural ODE arm's optimizer settings, selected on
# validation error at the final budget (sweep 2026-10-05-node-config: 3 datasets
# x warm start {0,1} x lr {1e-2,1e-3,1e-4}, seed 0, 6000 iterations; results in
# results/runs_python.csv). Selection is on validation error only, never test.
#
# The warm start is selected for DCR alone, the dataset whose readout holds most
# of the parameters: warm-starting fits the readout to a random initial latent,
# which pays only when the readout is too large for the optimizer to find
# otherwise. The MATLAB arm selects the same way, per dataset
# (getDatasetConfig.m).
DATASETS = {
    # nTrain/nVal of None means "use the split baked into the data file" (ELM).
    'ELM': dict(nTrain=None, nVal=None, nc=15, warm_start=0, lr=1e-2),
    'CDR': dict(nTrain=400,  nVal=200,  nc=8,  warm_start=0, lr=1e-2),   # Newman et al.
    'DCR': dict(nTrain=8000, nVal=1000, nc=16, warm_start=1, lr=1e-3),   # Newman et al.
}


def get(dataset):
    d = dataset.upper()
    if d not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dict(DATASETS[d], dataset=d, split_seed=SPLIT_SEED)
