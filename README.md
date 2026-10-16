# Weight-Parameterization in Continuous Time Deep Neural Networks for Surrogate Modeling

Manuscript, code, and data for:

```bibtex
@article{RossoRuthottoSargsyan,
  title         = {Weight-Parameterization in Continuous Time Deep Neural Networks for Surrogate Modeling},
  author        = {Haley Rosso and Lars Ruthotto and Khachik Sargsyan},
  year          = {2025},
  eprint        = {2507.22045},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2507.22045}
}
```

## Contents

```
main.tex       manuscript
matlab/        discretize-then-optimize experiments (ResNet, Hamiltonian network)
python/        optimize-then-discretize experiments (neural ODE) and the figure and table generators
data/          ELM and CDR datasets, dataset loaders, train/validation/test splits
results/       run ledgers, sweep manifests, curve data
tables/        generated tabular bodies, included by main.tex
figures/       generated figures, included by main.tex
```

Requirements: MATLAB R2022a or newer with the Symbolic Math Toolbox, Python 3.13 with
`python/requirements.txt`, and a LaTeX distribution with `latexmk`.

## Build

```bash
make test      # MATLAB and Python unit tests
make figures   # compile the figure sources to PDF
make tables    # regenerate tables/*.tex from results/
make paper     # main.pdf
```

`make tables`, `make figures` and `make paper` read the committed ledgers in `results/`
and do not retrain.

## Reproduce the experiments

Each sweep takes hours to days and writes to `results/raw/<sweepId>/`, which is not committed.

```bash
cd matlab/experiments && ./runNamedSweep.sh runPaperSweep paper 20
cd python && ./run_node_sweep.sh <sweepId>
```

[results/README.md](results/README.md) lists the sweeps behind each table and figure and the
format of the ledgers. [matlab/README.md](matlab/README.md) and
[python/README.md](python/README.md) describe the two pipelines.

## Data

- `data/NNERDS.mat` (ELM) and `data/CDR_Data.mat` (CDR) are included.
- `DCR_Data.mat` (64 MB) is hosted at [10.5281/zenodo.21537966](https://doi.org/10.5281/zenodo.21537966).
  `data/setupDCR.m` downloads it on first use. Batch jobs cannot answer its prompt, so run
  `setupDCR` once interactively, or place the file in `data/`, before launching a sweep.
- `data/splits.mat` holds the train/validation/test index permutations used by both pipelines.

## License

MIT, see [LICENSE](LICENSE). `matlab/meganet/` is forked from Meganet and keeps its own license.
