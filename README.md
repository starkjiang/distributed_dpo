# Distributed DPO — Empirical Validation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) -->

Empirical validation experiments for **FedDPO** and **DecDPO** from the paper  
*"Distributed Direct Preference Optimisation"*.

This repository reproduces the key convergence, ablation, and topology results across two human-preference datasets: **Anthropic/hh-rlhf** and **stanfordnlp/SHP**.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Experiment Design](#experiment-design)
  - [A. Comparative Study](#a-comparative-study)
  - [B. Ablation Studies](#b-ablation-studies)
- [Output Figures](#output-figures)
- [Design Decisions](#design-decisions)
- [Extending the Code](#extending-the-code)

---

## Overview

Direct Preference Optimisation (DPO) fine-tunes language models from human preference data without a separate reward model. This codebase validates two distributed variants:

| Algorithm | Description |
|-----------|-------------|
| **Centralized DPO** | Baseline — all data pooled on a single server |
| **FedDPO** | Federated averaging (FedAvg) with optional partial participation and gradient staleness |
| **DecDPO** | Fully decentralised gossip mixing over arbitrary communication graph topologies |

All experiments run on `distilgpt` (≈82 M parameters) for fast iteration. The framework is model-agnostic and straightforward to adapt to larger LLMs.

---

## Key Results

- **FedDPO (full participation)** matches centralized DPO convergence.
- **FedDPO (partial)** incurs a small variance floor that scales as `$\zeta^2_g/S`.
- **DecDPO** convergence depends on the *spectral gap* `$1-\rho^2` of the mixing matrix:  
  the consensus error satisfies `$\mathfrak{e}_\theta ∝ \eta^2 / (1 − \rho^2)` (Theorem 6.1).
- **Gradient staleness** increases the stationary gap roughly linearly in `q_{max}`.

---

## Project Structure

```
distributed-dpo/
├── main.py                        # Entry point
├── requirements.txt
├── configs/
│   └── default.yaml               # All hyperparameters
├── distributed_dpo/               # Core Python package
│   ├── __init__.py
│   ├── config.py                  # CFG dict, device, colour palettes
│   ├── data.py                    # Dataset loaders + agent partitioning
│   ├── models.py                  # Model init, tokenisation, DPO loss
│   ├── algorithms.py              # Centralized / FedDPO / DecDPO loops
│   ├── experiments.py             # Full experiment suite runner
│   ├── visualization.py           # All Matplotlib figure generators
│   └── utils.py                   # Helpers (smooth, savefig, JSON, …)
├── outputs/                       # (git-ignored) experiment artefacts
└── scripts/
    └── quick_test.sh              # Smoke-test: 5 rounds, hh-rlhf only
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/starkjiang/distributed-dpo.git
cd distributed-dpo
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note** — if you have a CUDA-capable GPU, install the matching PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Quick Start

### Run all experiments (both datasets)

```bash
python main.py
```

Results are saved to `./ddpo_results/`.

### Smoke-test (5 rounds, hh-rlhf only)

```bash
python main.py --rounds 5 --datasets hh-rlhf
```

### Custom output directory

```bash
python main.py --output ./my_experiment
```

### All CLI options

```
usage: main.py [-h] [--datasets {hh-rlhf,SHP} [{hh-rlhf,SHP} ...]]
               [--rounds ROUNDS] [--output OUTPUT] [--seed SEED]

optional arguments:
  --datasets   Datasets to run (default: hh-rlhf SHP)
  --rounds     Override comm_rounds (e.g. 5 for a quick smoke-test)
  --output     Override output directory (default: ./ddpo_results)
  --seed       Random seed (default: 42)
```

---

## Configuration

All hyperparameters live in two places:

| Location | Purpose |
|----------|---------|
| `configs/default.yaml` | Human-readable reference |
| `distributed_dpo/config.py` — `CFG` dict | Runtime config (read by all modules) |

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `distilroberta-base` | HuggingFace model ID |
| `num_agents` | `5` | Number of distributed agents |
| `comm_rounds` | `80` | Global communication rounds |
| `local_steps` | `3` | Local update steps per round (FedDPO) |
| `dec_local_steps` | `5` | Local update steps per round (DecDPO) |
| `beta` | `0.2` | DPO KL-penalty coefficient |
| `lr_adam` | `1e-4` | AdamW learning rate (FedDPO / Centralized) |
| `lr_sgd` | `3e-5` | SGD learning rate (fixed η) if using SGD |
| `participation` | `3` | Clients sampled per round (partial FedDPO) |
| `n_samples_agent` | `120` | Preference pairs per agent |
| `batch_size` | `4` | Mini-batch size |
| `grad_clip` | `1.0` | Max gradient norm |

To change a parameter, edit `distributed_dpo/config.py` directly or pass CLI flags for the subset supported by `main.py`.

---

## Experiment Design

### A. Comparative Study

Runs all four algorithms on the same data partition and plots gradient-norm and DPO-loss convergence.

| Tag | Algorithm |
|-----|-----------|
| `centralized` | Centralized DPO (AdamW, pooled data) |
| `feddpo_full` | FedDPO, all S=5 agents per round |
| `feddpo_partial` | FedDPO, S=3 agents sampled per round |
| `decdpo_ring` | DecDPO, ring topology, SGD |

### B. Ablation Studies

| Study | Values | Metric |
|-------|--------|--------|
| **B1** Local steps E | {1, 3, 6} | Gradient norm vs rounds |
| **B2** Participation S | {1, 3, 5} | Gradient norm vs rounds |
| **B3** Graph topology | path / ring / star / complete | DPO loss + consensus error vs spectral gap |
| **B4** Staleness q_max | {0, 2, 5} | Gradient norm vs rounds |

#### Graph topologies (B3)

| Topology | Spectral gap `$1−\rho^2$` | Expected consensus error |
|----------|---------------------|--------------------------|
| Path | lowest | highest |
| Ring | low | high |
| Star | medium | medium |
| Complete | highest | lowest |

The code empirically validates Theorem 6.1:  
**`$\mathfrak{e}_\theta ∝ \eta^2 / (1 − \rho^2)$`**

---

## Output Figures

After a full run, `ddpo_results/` contains:

```
ddpo_results/
├── fig_cross_dataset_comparison.{png,pdf}
├── fig_cross_dataset_loss.{png,pdf}
├── Anthropic-hh-rlhf/
│   ├── fig1_comparative.{png,pdf}
│   ├── fig2_local_steps.{png,pdf}
│   ├── fig3_participation.{png,pdf}
│   ├── fig4_topology.{png,pdf}
│   ├── fig5_staleness.{png,pdf}
│   ├── fig6_summary_dashboard.{png,pdf}
│   └── results.json
└── SHP/
    └── (same structure)
```

| Figure | Contents |
|--------|----------|
| `fig1_comparative` | Grad norm + DPO loss for all 4 algorithms |
| `fig2_local_steps` | Convergence vs E ∈ {1, 3, 6} |
| `fig3_participation` | Convergence vs S ∈ {1, 3, 5} |
| `fig4_topology` | Graph drawings + DPO loss + consensus-error scatter |
| `fig5_staleness` | Convergence vs q_max ∈ {0, 2, 5} |
| `fig6_summary_dashboard` | 2×3 overview of all experiments |
| `fig_cross_dataset_*` | Side-by-side hh-rlhf vs SHP comparison |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SGD for DecDPO** | Fixed step-size makes the spectral gap observable in convergence curves |
| **AdamW for FedDPO / Centralized** | More stable under non-IID preference distributions |
| **Token-sum log-prob** | Keeps DPO margins scale-correct across variable-length responses |
| **Gradient clipping (max-norm 1.0)** | Prevents exploding gradients in all three algorithms |
| **Consensus error tracked before + after gossip** | Isolates the per-round mixing improvement |
| **Metropolis–Hastings mixing weights** | Produces doubly-stochastic Π for any graph topology |

---

## Extending the Code

### Add a new dataset

1. Add a loader function to `distributed_dpo/data.py` returning `list[dict[str, str]]` with keys `"chosen"` and `"rejected"`.
2. Register it in the `DATASET_LOADERS` dict.

```python
def load_my_dataset() -> list[dict[str, str]]:
    ...

DATASET_LOADERS["my_dataset"] = load_my_dataset
```

### Add a new graph topology

Add an entry to the `graphs` dict in `algorithms.build_mixing_matrix`:

```python
graphs["grid"] = nx.grid_2d_graph(int(N**0.5), int(N**0.5))
```

### Swap the base model

Change `model_name` in `distributed_dpo/config.py`. Any HuggingFace causal LM works:

```python
CFG["model_name"] = "gpt2"          # ~117 M params
CFG["model_name"] = "facebook/opt-125m"
```

### Run a single algorithm programmatically

```python
from distributed_dpo import init_models, run_fed_dpo
from distributed_dpo.config import CFG

init_models()
agent_data = { ... }  # your data
history = run_fed_dpo(agent_data, comm_rounds=20, local_steps=3, beta=0.1)
```

---

## License

MIT — see [LICENSE](LICENSE).
