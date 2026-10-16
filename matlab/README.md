# MATLAB: discretize-then-optimize experiments

Trains the ResNet and Hamiltonian networks and produces the ResNet-results table in the appendix
(`tables/tableC1.tex`), the weight-trajectory data, and the ledger `results/runs.csv`.

```
meganet/       network layers and optimizers, forked from Meganet
experiments/   training function and the sweep drivers behind the paper
lib/           dataset settings, run naming, sweep manifests, provenance checks
analysis/      ledger collection, table and figure data, data splits
tests/         unit tests
```

## Requirements

MATLAB R2022a or newer. The Symbolic Math Toolbox is needed because the `'Legendre'` basis in
`meganet/getPolynomialBasis.m` calls `legendreP`.

## Data

`data/NNERDS.mat` (ELM) and `data/CDR_Data.mat` (CDR) are included. `data/DCR_Data.mat` is hosted
at [10.5281/zenodo.21537966](https://doi.org/10.5281/zenodo.21537966); `data/setupDCR.m` downloads it
on first use, or place it in `data/` yourself.

## Run

From `experiments/`:

```matlab
addpath(genpath('..'))
runExperiment('hamiltonian', 1, 3, 'sgd', 'Legendre', 'DCR')   % one configuration
```

The arguments of `runExperiment` are, in order: `dynamic` (`'ResNN'`, `'hamiltonian'`), terminal time `T`, basis degree, optimizer (`'sgd'` for ADAM, `'GNvpro'`), `basis`
(`'monomial'`, `'Legendre'`, `'none'`), dataset (`'ELM'`, `'CDR'`, `'DCR'`), and an optional seed
(default 20). The train/validation/test split is fixed by `rng(42)` and does not depend on the seed.

The sweeps behind the paper are launched from a shell with `runNamedSweep.sh <driver> <slug> [workers]`:

| driver | sweep |
|---|---|
| `runPaperSweep` | every configuration in the ResNet results: bases, degrees 0 to 11, horizons, optimizers, ten seeds |
| `runGnvproSeeds` | GNvpro seeds 30 to 59 on the degree axis |
| `runConvergenceSweep` | degrees 0 and 11 and the unconstrained model at five times the ADAM budget, CDR and DCR |
| `runPrecisionMonomialSweep` | monomial basis, degrees 10 and 11, single against double precision, CDR |

Each sweep writes one `.mat` file per run to `results/raw/<sweepId>/matlab/`, and a manifest to
`results/sweeps/<sweepId>.json`.

## Collect and tabulate

```matlab
% from analysis/, after addpath(genpath('..'))
collectResults    % results/raw/ -> results/runs.csv
makeTableC1      % results/runs.csv -> tables/tableC1.tex
```

`collectResults` refuses a ledger that mixes sweeps or commits (`checkProvenance.m`). The ledger
carries three parameter counts: `nParamsDynamics` (the dynamics weights only; at degree 3 it is 960
on ELM, 288 on CDR and 1088 on DCR), `nParamsTheta` (adds the input block), and `nParamsTotal` (adds the readout). Use
`nParamsDynamics` for the tables.

`exportWeightTrajectories` writes `results/weight_trajectories.mat`, which
`python/export_weight_figure.py` plots.
