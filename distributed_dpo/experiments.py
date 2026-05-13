"""
distributed_dpo/experiments.py
================================
Orchestrates the full experiment suite for a single dataset:

  A. Comparative study   — Centralized vs FedDPO (full/partial) vs DecDPO ring
  B. Ablation studies
       B1. Local steps  E  ∈ {1, 3, 6}
       B2. Participation S ∈ {1, 3, 5}
       B3. Graph topology — path / ring / star / complete
       B4. Staleness      q_max ∈ {0, 2, 5}
"""

from typing import Any

from .config import CFG
from .algorithms import (
    run_centralized_dpo,
    run_fed_dpo,
    run_dec_dpo,
)

# ── Types ─────────────────────────────────────────────────────────────────────
AgentData    = dict[int, list[dict[str, str]]]
ExperimentResult = tuple[
    dict[str, list],       # rc  — comparative histories
    dict[str, Any],        # ra  — ablation histories
    dict[str, dict],       # spectral_gaps
    dict[str, str],        # labels
    float,                 # rho_ring
]


def run_experiments(agent_data: AgentData, dataset_name: str) -> ExperimentResult:
    """
    Run the full experiment suite on ``agent_data`` and return all results.

    Parameters
    ----------
    agent_data   : per-agent list of {chosen, rejected} dicts
    dataset_name : display name used in progress bars and print output

    Returns
    -------
    rc, ra, spectral_gaps, labels, rho_ring
    """
    print(f"\n{'=' * 60}")
    print(f"  Running experiments on: {dataset_name}")
    print(f"{'=' * 60}")

    N  = CFG["num_agents"]
    R  = CFG["comm_rounds"]
    E  = CFG["local_steps"]
    bt = CFG["beta"]

    # ── A. Comparative study ──────────────────────────────────────────────────
    print("\n[A] Comparative Study")
    rc: dict[str, list] = {}

    print("  [A1] Centralized DPO")
    rc["centralized"] = run_centralized_dpo(agent_data, R, E, bt)

    print("  [A2] FedDPO — full participation")
    rc["feddpo_full"] = run_fed_dpo(agent_data, R, E, bt)

    print("  [A3] FedDPO — partial participation")
    rc["feddpo_partial"] = run_fed_dpo(
        agent_data, R, E, bt, participation=CFG["participation"]
    )

    print("  [A4] DecDPO — ring topology")
    h_dec, rho_ring, _ = run_dec_dpo(agent_data, R, bt, graph_type="ring")
    rc["decdpo_ring"] = h_dec

    labels = {
        "centralized":    "Centralized DPO",
        "feddpo_full":    f"FedDPO (full, S={N})",
        "feddpo_partial": f"FedDPO (partial, S={CFG['participation']})",
        "decdpo_ring":    f"DecDPO ring (ρ={rho_ring:.2f})",
    }

    # ── B. Ablation studies ───────────────────────────────────────────────────
    print("\n[B] Ablation Studies")
    ra: dict[str, Any] = {}

    # B1 — local steps E ∈ {1, 3, 6}
    print("  [B1] Local steps E ∈ {1, 3, 6}")
    ra["local_steps"] = {}
    for E_val in [1, 3, 6]:
        ra["local_steps"][str(E_val)] = run_fed_dpo(agent_data, R, E_val, bt)

    # B2 — participation S ∈ {1, 3, 5}
    print("  [B2] Participation S ∈ {1, 3, 5}")
    ra["participation"] = {}
    for S_val in [1, 3, 5]:
        ra["participation"][str(S_val)] = run_fed_dpo(
            agent_data, R, E, bt, participation=S_val
        )

    # B3 — graph topology
    print("  [B3] Graph topology — path / ring / star / complete")
    ra["graph_topology"] = {}
    spectral_gaps: dict[str, dict] = {}
    for gt in ["path", "ring", "star", "complete"]:
        h, rho, gap = run_dec_dpo(agent_data, R, bt, graph_type=gt)
        ra["graph_topology"][gt] = h
        spectral_gaps[gt] = {"rho": float(rho), "spectral_gap": float(gap)}

    # B4 — staleness q_max ∈ {0, 2, 5}
    print("  [B4] Staleness q_max ∈ {0, 2, 5}")
    ra["staleness"] = {}
    for qmax in [0, 2, 5]:
        ra["staleness"][str(qmax)] = run_fed_dpo(
            agent_data, R, E, bt,
            participation=CFG["participation"],
            staleness=qmax,
        )

    return rc, ra, spectral_gaps, labels, rho_ring
