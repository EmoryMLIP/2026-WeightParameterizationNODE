"""The one sweep the manuscript draws from, and which of its runs are the defaults.

Every table and figure generator imports these instead of naming sweeps itself.

The paper sweep also contains a readout warm-start ablation: runs at Legendre
d=3, T=1, Adam that flip each dataset's configured warmStartW. They share every
ledger column with the default runs except warmStartW, so a generator that does
not filter on it averages two initializations into one cell.
"""

PAPER_SWEEP = '2026-10-02-paper'

# Per-dataset default, as selected on validation error in getDatasetConfig.m.
DEFAULT_WARMSTART = {'ELM': True, 'DCR': True, 'CDR': False}


def is_default_run(row):
    """True unless the row is from the warm-start ablation.

    A row with no warmStartW value used the dataset default.
    """
    v = (row.get('warmStartW') or '').strip().lower()
    if v in ('', 'nan'):
        return True
    return (v in ('1', '1.0', 'true')) == DEFAULT_WARMSTART[row['dataset']]


# GNvpro seeds 30-59 on the T=1 degree axis, a declared top-up of PAPER_SWEEP.
# GNvpro's seed-to-seed spread is wide and its runs cost seconds, so the paired
# comparison against the 10% margin gets forty seeds per GNvpro cell and ten per Adam cell.
GNVPRO_TOPUP = '2026-10-04-gnvpro-seeds'


def degree_axis_sweeps(opti):
    """Sweeps whose runs form one T=1 degree-axis cell for this optimizer."""
    return {PAPER_SWEEP, GNVPRO_TOPUP} if opti == 'GNvpro' else {PAPER_SWEEP}
