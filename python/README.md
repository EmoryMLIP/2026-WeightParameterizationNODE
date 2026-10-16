# Python: optimize-then-discretize experiments and generators

Trains the neural ODE, and generates every figure and table of the manuscript from the committed
ledgers in `results/`.

## Requirements

Python 3.13.7 and `requirements.txt`. The sweep scripts call `python/.venv/bin/python`, so install into a
virtual environment there (see the header of `requirements.txt`).

## Neural ODE

```
timeparam.py                    time-dependent weight parameterization
dataset_config.py               per-dataset settings, mirrors lib/getDatasetConfig.m
surrogate/train_surrogate_node.py   one training run
run_node_sweep.sh <sweepId>     the sweep behind the neural ODE table
collect_python.py               results/raw/ -> results/runs_python.csv
```

Gradients use the continuous adjoint (`torchdiffeq.odeint_adjoint`). The flag
`--backprop-through-solver` selects the discretize-then-optimize gradient instead. Each run records
its choice under the `gradient` key.

The selection of learning rate and warm start (`run_config_selection.sh`) is a separate sweep with
the same interface as `run_node_sweep.sh`.

## Tables and figures

| script | output | manuscript label |
|---|---|---|
| `make_degree_efficiency_table.py` | `tables/table2.tex` | `tab:degree_efficiency` |
| `make_tables.py` | `tables/table3.tex` | `tab:meanstd` |
| `export_effect_ledger.py` | `figures/effect_ledger.tex` | `fig:effect_ledger` |
| `export_paired_panels.py` | `figures/paired_panels.tex` | `fig:paired_panels` |
| `export_lever_figure.py` | `figures/degree_lever.tex` | `fig:degree_lever` |
| `export_weight_figure.py` | `figures/weight_trajectories.tex` | `fig:weight_trajectories` |
| `surrogate/aggregate_fig4_traces.py`, `surrogate/build_fig4_averaged.py` | `figures/node_nfe_{elm,cdr,dcr}.pdf` | `fig:neuralODE_loss` |

`aggregate_fig4_traces.py --sweep 2026-10-05-node-final` writes `results/traces/<sweepId>/node_traces.json`,
and `build_fig4_averaged.py --sweep 2026-10-05-node-final` plots it.

`paired_stats.py` implements the paired tests the generators share. `paper_statistics.py` prints
the aggregate statistics quoted in the text, and `audit_numbers.py` checks that each number in the
results sections appears in a ledger.

## Tests

```bash
python3 test_timeparam.py    # basis, Gram matrix, adjoint against backpropagation, Taylor rate
python3 test_paired_stats.py
```
