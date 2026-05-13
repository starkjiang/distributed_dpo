"""
distributed_dpo
================
Empirical validation experiments for Federated and Decentralised
Direct Preference Optimisation (FedDPO / DecDPO).

Package layout
--------------
config          — hyperparameters, colour palettes, device selection
data            — dataset loaders (hh-rlhf, SHP) and agent partitioning
models          — model init, DPO loss, optimiser factories
algorithms      — Centralized DPO, FedDPO, DecDPO training loops
experiments     — full experiment suite runner
visualization   — all Matplotlib figure generators
utils           — helpers (smooth, savefig, JSON serialisation, etc.)
"""

from .config       import CFG, DEVICE, SEED, MULTI_SEEDS
from .data         import DATASET_LOADERS, partition_to_agents
from .models       import init_models, make_model, dpo_loss_batch
from .algorithms   import run_centralized_dpo, run_fed_dpo, run_dec_dpo
from .experiments  import run_experiments
from .visualization import (
    plot_comparative,
    plot_comparative_multiseed,
    plot_local_steps,
    plot_local_steps_extended,
    plot_participation,
    plot_topology,
    plot_staleness,
    plot_summary_dashboard,
    plot_cross_dataset,
    plot_cross_dataset_loss,
)
from .utils import (
    smooth,
    get,
    savefig,
    ds_tag,
    to_json,
    save_results_json,
    print_summary,
    list_output_files,
)

__all__ = [
    "CFG", "DEVICE", "SEED", "MULTI_SEEDS",
    "DATASET_LOADERS", "partition_to_agents",
    "init_models", "make_model", "dpo_loss_batch",
    "run_centralized_dpo", "run_fed_dpo", "run_dec_dpo",
    "run_experiments",
    "plot_comparative", "plot_comparative_multiseed",
    "plot_local_steps", "plot_local_steps_extended",
    "plot_participation",
    "plot_topology", "plot_staleness", "plot_summary_dashboard",
    "plot_cross_dataset", "plot_cross_dataset_loss",
    "smooth", "get", "savefig", "ds_tag", "to_json",
    "save_results_json", "print_summary", "list_output_files",
]
