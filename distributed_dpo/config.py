"""
distributed_dpo/config.py
==========================
Central configuration for all Distributed DPO experiments.
All hyperparameters, paths, and plot styles live here.
"""

import os
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
MULTI_SEEDS = [42, 43, 44]

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Experiment hyperparameters ────────────────────────────────────────────────
CFG = dict(
    model_name      = "distilgpt2",
    num_agents      = 3,
    max_length      = 64,
    batch_size      = 4,
    beta            = 0.2,
    lr_adam         = 1e-4,        # AdamW lr — FedDPO / Centralized
    lr_sgd          = 3e-5,        # SGD lr  — DecDPO (fixed η reveals topology)
    grad_clip       = 1.0,
    comm_rounds     = 5,
    local_steps     = 1,
    dec_local_steps = 1,           # more steps before mixing → topology matters more
    participation   = 2,
    n_samples_agent = 120,
    output_dir      = "./ddpo_results",
)

# ── Algorithm colour palette ──────────────────────────────────────────────────
COLORS = {
    "centralized":    "#2C7BB6",
    "feddpo_full":    "#1A9641",
    "feddpo_partial": "#F48022",
    "decdpo_ring":    "#D7191C",
}

# ── Graph-topology colour palette ─────────────────────────────────────────────
GT_COLORS = {
    "path":     "#E63946",
    "ring":     "#2A9D8F",
    "star":     "#E9C46A",
    "complete": "#264653",
}

# ── Dataset colour palette ────────────────────────────────────────────────────
DS_COLORS = {
    "hh-rlhf": "#2C7BB6",
    "SHP":     "#D7191C",
}

# ── Matplotlib defaults ───────────────────────────────────────────────────────
PLOT_RCPARAMS = {
    "font.size":           16,
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.35,
    "figure.dpi":          130,
}


def ensure_output_dir(subdir: str | None = None) -> str:
    """Create and return the (sub-)directory for saving outputs."""
    path = os.path.join(CFG["output_dir"], subdir) if subdir else CFG["output_dir"]
    os.makedirs(path, exist_ok=True)
    return path
