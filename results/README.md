# Results

Every number in the manuscript is produced by a script in this repository from the ledgers
below. `tables/*.tex` hold only a tabular body; `main.tex` supplies caption and label.

```
run  ->  per-run record  ->  ledger  ->  tables/*.tex, figures/*.tex  ->  main.pdf
```

| | MATLAB | Python |
|---|---|---|
| run | `runExperiment.m` | `train_surrogate_node.py` |
| per-run record | `results/raw/<sweepId>/matlab/<tag>.mat` | `results/raw/<sweepId>/python/<tag>.json` |
| ledger | `results/runs.csv` | `results/runs_python.csv` |

## Files in `results/`

| path | committed | content |
|---|---|---|
| `runs.csv` | yes | MATLAB ledger, one row per run |
| `runs_python.csv` | yes | Python ledger, one row per run |
| `sweeps/<sweepId>.json` | yes | manifest: commit, status, run counts |
| `traces/<sweepId>/` | yes | curves plotted in `fig:neuralODE_loss` |
| `weight_trajectories.mat` | yes | data for `fig:weight_trajectories` |
| `raw/<sweepId>/` | no | trained weights, per-run records, logs |

## Sweeps

A sweep is a set of runs made at one commit, identified by `<sweepId>` = `YYYY-MM-DD-slug`. A table
draws on one sweep, or on a sweep and a declared extension of it. `checkProvenance.m` refuses a
ledger that spans commits, has runs without a recorded commit, or has uneven seed coverage.

A manifest with `status: "running"` belongs to a crashed or killed sweep and must not be used.

Run tags have the form `<dataset>_<arch>_<basis>_d<D>_T<T>_<opti>_s<seed>`, for example
`DCR_ResNN_Legendre_d3_T1_GNvpro_s20`.

| sweep | runs | content |
|---|---|---|
| `2026-10-02-paper` | 3,860 | all discretize-then-optimize runs: bases, degrees 0 to 11, unconstrained model, horizons, unscaled basis, single precision, regularization and warm-start selection; seeds 20 to 29 |
| `2026-10-04-gnvpro-seeds` | 2,340 | GNvpro seeds 30 to 59 on the degree axis at T=1 |
| `2026-10-04-precision-monomial` | 160 | monomial basis, CDR, degrees 10 and 11, single and double precision |
| `2026-10-05-convergence` | 60 | degrees 0 and 11 and the unconstrained model at five times the ADAM budget |
| `2026-10-05-node-final` | 60 | neural ODE table, five seeds |
| `2026-10-05-node-config` | 18 | neural ODE learning rate and warm start, selected on validation error |

All sweeps ran on one host (euler, MATLAB R2022a, Python 3.13.7 with torch 2.8.0 on CPU). The
MATLAB sweeps reproduce an earlier run of the same sweeps bit for bit on that host. On a second
host (zuber) double-precision runs agreed to 1e-13 (GNvpro) or better, but single-precision GNvpro
runs with the monomial basis at degrees 10 and 11 differed by up to a factor of three per seed.
Python runs differ in the last digits between platforms.

The MATLAB pipeline trains on raw targets and the Python pipeline on z-scored targets, so absolute
errors in the two pipelines are not comparable.

## Reading the ledgers

`runs.csv` (MATLAB) and `runs_python.csv` (neural ODE) hold one row per run. Columns of `runs.csv`:

| column | meaning |
|---|---|
| `sweep`, `commit` | sweep ID and the commit that ran it |
| `dataset`, `dynamic`, `basis`, `d`, `T`, `opti`, `seed` | configuration: dataset, `ResNN` or `hamiltonian`, `Legendre`, `monomial`, `monomial-raw` or `none`, degree (0 for `none`), horizon, `sgd` (Adam) or `GNvpro`, seed |
| `precision`, `alpha1`, `alpha2`, `maxEpochs`, `warmStartW` | settings that differ between runs of one sweep |
| `nTrain`, `nVal`, `nTest`, `nc`, `nt` | split sizes, channel width, time steps |
| `bestEpoch`, `elapsedSeconds` | epoch of the best validation iterate, wall time |
| `train_*`, `val_*`, `test_*` | error at that iterate; suffix `agg` is the aggregate relative error, `med` and `mean` are per-sample statistics |
| `trivial_test_agg`, `trivial_test_med` | error of the training-mean predictor |
| `nParamsDynamics`, `nParamsTheta`, `nParamsTotal` | parameter counts |

`runs_python.csv` has the same configuration columns plus `gradient`, `alpha`, `niters`, `bestIter`,
`train_nfe`, `val_nfe`, `test_nfe` (function evaluations) and `nParamsTotal`. Use `test_agg`.

The paper sweep also holds runs that are not defaults: the alpha sweep (`alpha1` other than 1e-10),
single precision, and the warm-start flip (`warmStartW` opposite to the dataset default, 1 for ELM
and DCR and 0 for CDR). Filter them out before averaging:

```python
import numpy as np, pandas as pd
d = pd.read_csv('results/runs.csv')
default_ws = {'ELM': 1, 'DCR': 1, 'CDR': 0}
p = d[(d.sweep == '2026-10-02-paper') & (d.alpha1 == 1e-10) & (d.precision == 'double')
      & (d.warmStartW == d.dataset.map(default_ws))]
cell = p[(p.dataset == 'CDR') & (p.dynamic == 'ResNN') & (p.opti == 'sgd') & (p['T'] == 1)]
gm = lambda x: np.exp(np.log(x).mean())            # geometric mean over seeds
print(cell.groupby(['basis', 'd']).test_agg.agg(['count', gm]))
```

Averages are geometric means over seeds. Comparisons pair two configurations by `seed`, as in
`python/paired_stats.py`.

## Statistics

Comparisons are paired by seed on log test error (`python/paired_stats.py`). A seed fixes the
initialization and minibatch order of every configuration run under it. A difference is claimed
when the 95% interval of the ratio excludes 1, similarity within 10% when the whole interval lies
inside [1/1.1, 1.1], with no multiple-comparison correction; trends along the degree axis use Page's
L test. `paper_statistics.py` prints every family the text quotes.

Two quantities accompany each comparison. `mde(r)` is the smallest ratio the comparison could
resolve at 80% power and level 0.05. `seed_gsd(cell)` is the geometric standard deviation of one
configuration across seeds.

## Python per-run record

| key | meaning |
|---|---|
| `dataset`, `basis`, `degree`, `seed`, `nc`, `T` | configuration |
| `gradient` | `continuous_adjoint` or `backprop_through_solver` |
| `split_seed` | fixed; the split does not vary with `seed` |
| `train_relerr`, `val_relerr`, `test_relerr` | aggregate relative error |
| `train_nfe`, `val_nfe`, `test_nfe` | function evaluations per region |
| `trivial_relerr` | error of the training-mean predictor |
| `best_iter`, `stopped_early` | model selection |
| `trainable_params` | parameters with `requires_grad` |

The MATLAB ledger carries the same information plus `test_agg`, `test_med`, `test_mean` (three error
conventions) and `nParamsDynamics`, `nParamsTheta`, `nParamsTotal` (three parameter counts). The
tables use `test_agg` and `nParamsDynamics`.

## Metric

The error is the aggregate relative error `||Pred - C||_F / ||C||_F`, evaluated at the
best-validation checkpoint. On mean-centred targets (DCR) the training-mean predictor scores 1.0, so
a value above 1 is worse than predicting the mean. `trivialPredictorError.m` computes this
reference.

## Provenance

Every generated file begins with a comment naming the commit, timestamp, interpreter version and
host (`provenance.m`, `provenance.py`). A stamp reading `(WORKING TREE DIRTY)` marks an artifact
built from uncommitted code.

The sweeps ran before this repository's history was consolidated into its first commit. The code
they ran is byte-identical to that commit on every path in `computational-paths.txt`, so the
ledgers, manifests and stamps name it, and the run dates in the manifests precede its commit date.
