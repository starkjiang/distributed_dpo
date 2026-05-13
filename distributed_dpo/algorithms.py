"""
distributed_dpo/algorithms.py
==============================
Training algorithms:
  - Centralized DPO         — pooled data, single model, AdamW
  - FedDPO                  — federated averaging, optional partial participation
                              and gradient staleness
  - DecDPO                  — decentralised gossip mixing over configurable
                              communication graphs

Key design choices
------------------
- DecDPO uses SGD with a fixed step-size so the spectral gap has a
  visible effect on convergence and consensus error.
- FedDPO / Centralized use AdamW (more stable for non-IID preferences).
- Gradient clipping (max-norm 1.0) is applied everywhere.
- DecDPO tracks consensus error  e_θ = (1/N) Σ_i ||θ_i − θ̄||²
  both *before* and *after* the gossip mixing step.
"""

import copy
import random
from typing import Any

import numpy as np
import networkx as nx
import torch

from tqdm.auto import tqdm

from .config import CFG, DEVICE
from .models import (
    make_model,
    make_adam,
    make_sgd,
    dpo_loss_batch,
    safe_float,
    grad_norm_sq,
)

# ── Type alias ────────────────────────────────────────────────────────────────
History = list[dict[str, Any]]


# ── Shared helper ─────────────────────────────────────────────────────────────

def sample_batch(data: list, bs: int) -> list:
    return random.sample(data, min(bs, len(data)))


def consensus_error(agents: list) -> float:
    """
    Compute (1/N) Σ_i ||θ_i − θ̄||²

    θ̄ is the element-wise mean of all agent state-dicts.
    """
    N = len(agents)
    mean_sd = copy.deepcopy(agents[0].state_dict())

    for key in mean_sd:
        if mean_sd[key].dtype.is_floating_point:
            mean_sd[key] = sum(a.state_dict()[key].float() for a in agents) / N

    err = 0.0
    for a in agents:
        for key, val in a.state_dict().items():
            if val.dtype.is_floating_point:
                err += (val.float() - mean_sd[key]).norm(2).item() ** 2
    return err / N


# =============================================================================
# A. Centralized DPO
# =============================================================================

def run_centralized_dpo(
    agent_data: dict[int, list],
    comm_rounds: int,
    local_steps: int,
    beta: float,
) -> History:
    """
    Baseline: all agents' data pooled into a single dataset, trained with
    AdamW on a single model.
    """
    pooled = [item for d in agent_data.values() for item in d]
    model  = make_model()
    opt    = make_adam(model)
    history: History = []

    for r in tqdm(range(comm_rounds), desc="Centralized DPO", leave=False):
        gn = loss_acc = 0.0
        for _ in range(local_steps):
            opt.zero_grad()
            loss = dpo_loss_batch(model, sample_batch(pooled, CFG["batch_size"]), beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            gn       += grad_norm_sq(model)
            loss_acc += safe_float(loss)
            opt.step()

        history.append({
            "round":        r,
            "grad_norm_sq": gn / local_steps,
            "loss":         loss_acc / local_steps,
        })

    del model
    torch.cuda.empty_cache()
    return history


# =============================================================================
# B. FedDPO  (Federated Averaging)
# =============================================================================

def run_fed_dpo(
    agent_data:    dict[int, list],
    comm_rounds:   int,
    local_steps:   int,
    beta:          float,
    participation: int | None = None,
    staleness:     int        = 0,
) -> History:
    """
    FedAvg-style Distributed DPO.

    Parameters
    ----------
    participation : int | None
        Number of agents sampled per round (``None`` → all agents).
    staleness : int
        Maximum gradient staleness  q_max.  0 = no staleness.
        Each agent starts from its own stale_buffer copy instead of the
        latest global model.
    """
    N = len(agent_data)
    S = participation or N

    global_sd    = copy.deepcopy(make_model().state_dict())
    stale_buffer = {i: copy.deepcopy(global_sd) for i in range(N)}
    history: History = []

    desc = f"FedDPO S={S} E={local_steps} stale={staleness}"
    for r in tqdm(range(comm_rounds), desc=desc, leave=False):
        selected = random.sample(range(N), min(S, N))
        local_params, gn, loss_acc = [], 0.0, 0.0

        for i in selected:
            start_sd = stale_buffer[i] if staleness > 0 else global_sd
            lm = make_model()
            lm.load_state_dict(copy.deepcopy(start_sd))
            lm.train()
            opt = make_adam(lm)

            for _ in range(local_steps):
                opt.zero_grad()
                loss = dpo_loss_batch(
                    lm,
                    sample_batch(agent_data[i], CFG["batch_size"]),
                    beta,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lm.parameters(), CFG["grad_clip"])
                gn       += grad_norm_sq(lm)
                loss_acc += safe_float(loss)
                opt.step()

            local_params.append(copy.deepcopy(lm.state_dict()))
            stale_buffer[i] = copy.deepcopy(global_sd)
            del lm
            torch.cuda.empty_cache()

        # FedAvg aggregation
        new_sd = copy.deepcopy(local_params[0])
        for key in new_sd:
            if new_sd[key].dtype.is_floating_point:
                for j in range(1, len(local_params)):
                    new_sd[key] = new_sd[key] + local_params[j][key]
                new_sd[key] = new_sd[key] / len(local_params)
        global_sd = new_sd

        n_steps = len(selected) * local_steps
        history.append({
            "round":        r,
            "grad_norm_sq": gn / n_steps,
            "loss":         loss_acc / n_steps,
        })

    torch.cuda.empty_cache()
    return history


# =============================================================================
# C. DecDPO  (Decentralised / Gossip)
# =============================================================================

def build_mixing_matrix(
    graph_type: str,
    N: int,
) -> tuple[np.ndarray, float, float]:
    """
    Build a doubly-stochastic Metropolis–Hastings mixing matrix for the
    requested graph topology and return (Π, ρ, 1−ρ²).

    Supported topologies: "path", "ring", "star", "complete".
    """
    graphs = {
        "path":     nx.path_graph(N),
        "ring":     nx.cycle_graph(N),
        "star":     nx.star_graph(N - 1),
        "complete": nx.complete_graph(N),
    }
    G  = graphs[graph_type]
    Pi = np.zeros((N, N))

    for i, j in G.edges():
        w = 1.0 / (1 + max(G.degree(i), G.degree(j)))
        Pi[i, j] = Pi[j, i] = w

    for i in range(N):
        Pi[i, i] = 1.0 - Pi[i, :].sum() + Pi[i, i]

    eigvals  = np.sort(np.abs(np.linalg.eigvalsh(Pi)))[::-1]
    rho      = float(eigvals[1])
    spec_gap = 1.0 - rho ** 2
    print(f"    [{graph_type}] ρ={rho:.4f}, 1-ρ²={spec_gap:.4f}")
    return Pi, rho, spec_gap


def run_dec_dpo(
    agent_data:  dict[int, list],
    comm_rounds: int,
    beta:        float,
    graph_type:  str = "ring",
) -> tuple[History, float, float]:
    """
    Decentralised DPO with gossip mixing.

    Uses SGD with a fixed step-size so the spectral gap of the mixing
    matrix has a visible effect on both convergence and consensus error.

    Tracks per-round:
      - DPO loss (convergence error proxy)
      - consensus error e_θ before and after gossip mixing
      - spectral gap 1−ρ² for the chosen topology
    """
    N = len(agent_data)
    Pi, rho, spec_gap = build_mixing_matrix(graph_type, N)
    local_steps = CFG["dec_local_steps"]

    agents = [make_model() for _ in range(N)]
    opts   = [make_adam(a)  for a in agents]
    history: History = []

    for r in tqdm(range(comm_rounds), desc=f"DecDPO [{graph_type}]", leave=False):
        gn = loss_acc = 0.0

        # ── Local Adam steps ───────────────────────────────────────────────
        for i in range(N):
            agents[i].train()
            for _ in range(local_steps):
                opts[i].zero_grad()
                loss = dpo_loss_batch(
                    agents[i],
                    sample_batch(agent_data[i], CFG["batch_size"]),
                    beta,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agents[i].parameters(), CFG["grad_clip"])
                gn       += grad_norm_sq(agents[i])
                loss_acc += safe_float(loss)
                opts[i].step()

        ce_before = consensus_error(agents)

        # ── Gossip mixing step ────────────────────────────────────────────
        old_sds = [copy.deepcopy(a.state_dict()) for a in agents]
        for i in range(N):
            new_sd = copy.deepcopy(old_sds[i])
            for key in new_sd:
                if new_sd[key].dtype.is_floating_point:
                    new_sd[key] = sum(
                        Pi[i, j] * old_sds[j][key].float() for j in range(N)
                    )
            agents[i].load_state_dict(new_sd)
            agents[i].train()

        ce_after = consensus_error(agents)
        n_steps  = N * local_steps

        history.append({
            "round":            r,
            "grad_norm_sq":     gn / n_steps,
            "loss":             loss_acc / n_steps,
            "consensus_before": ce_before,
            "consensus_after":  ce_after,
            "rho":              rho,
            "spec_gap":         spec_gap,
        })

    for a in agents:
        del a
    torch.cuda.empty_cache()
    return history, rho, spec_gap
