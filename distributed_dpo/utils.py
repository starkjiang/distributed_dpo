"""
distributed_dpo/utils.py
=========================
Shared utility functions used across the project.
"""

import json
import os
from typing import Any

import numpy as np
import torch

from .config import CFG


# ── History helpers ───────────────────────────────────────────────────────────

def get(history: list[dict], key: str) -> list:
    """Extract a scalar series from a list of per-round dicts."""
    return [h[key] for h in history]


def smooth(vals: list[float], w: int = 5) -> list[float]:
    """Trailing-window moving average (causal, no lookahead)."""
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        out.append(float(np.mean(vals[lo:i + 1])))
    return out


# ── I/O helpers ───────────────────────────────────────────────────────────────

def savefig(fig, name: str, subdir: str | None = None) -> None:
    """Save a matplotlib figure as both .png and .pdf."""
    d = os.path.join(CFG["output_dir"], subdir) if subdir else CFG["output_dir"]
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight")
    print(f"    Saved {name}.png/.pdf")


def ds_tag(ds_name: str) -> str:
    """Convert a dataset name to a filesystem-safe subdirectory tag."""
    return ds_name.replace("/", "-")


def to_json(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.item()
    return obj


def save_results_json(
    rc:            dict,
    ra:            dict,
    spectral_gaps: dict,
    ds_name:       str,
    subdir:        str,
) -> None:
    """Serialise experiment results to JSON."""
    out_path = os.path.join(CFG["output_dir"], subdir, "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "config":        CFG,
        "dataset":       ds_name,
        "comparative":   rc,
        "ablation":      ra,
        "spectral_gaps": spectral_gaps,
    }
    with open(out_path, "w") as f:
        json.dump(to_json(payload), f, indent=4)
    print(f"    Saved results.json → {out_path}")


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(
    rc:            dict,
    ra:            dict,
    spectral_gaps: dict,
    labels:        dict[str, str],
    ds_name:       str,
) -> None:
    """Print a concise per-dataset results table to stdout."""
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {ds_name}")
    print(f"{'=' * 70}")

    print(f"\n{'Algorithm':<30} {'Final GN²':>12} {'Final Loss':>12}")
    print("-" * 56)
    for key, label in labels.items():
        gns = get(rc[key], "grad_norm_sq")
        ls  = get(rc[key], "loss")
        print(f"{label:<30} {np.mean(gns[-5:]):>12.5f} {np.mean(ls[-5:]):>12.5f}")

    print(f"\n{'Topology':<12} {'ρ':>8} {'1−ρ²':>8} {'Final e_θ':>12} {'Final Loss':>12}")
    print("-" * 56)
    for gt in ["path", "ring", "star", "complete"]:
        rho_v = spectral_gaps[gt]["rho"]
        gap_v = spectral_gaps[gt]["spectral_gap"]
        ce    = float(np.mean(get(ra["graph_topology"][gt], "consensus_after")[-5:]))
        loss  = float(np.mean(get(ra["graph_topology"][gt], "loss")[-5:]))
        print(f"{gt:<12} {rho_v:>8.4f} {gap_v:>8.4f} {ce:>12.6f} {loss:>12.5f}")


def list_output_files(output_dir: str) -> None:
    """Walk the output directory and print a size-annotated file tree."""
    print(f"\n{'=' * 60}")
    print(f"✅  Done.  All outputs in: {output_dir}/")
    print(f"{'=' * 60}")
    for root, dirs, files in os.walk(output_dir):
        dirs.sort()
        files.sort()
        level  = root.replace(output_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for fname in files:
            sz = os.path.getsize(os.path.join(root, fname)) / 1024
            print(f"{indent}  {fname:<45} {sz:>7.1f} KB")
