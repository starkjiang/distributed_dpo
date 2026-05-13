"""
distributed_dpo/data.py
========================
Dataset loaders and per-agent data partitioning.

Supported datasets
------------------
- Anthropic/hh-rlhf  — human helpfulness & harmlessness preference pairs
- stanfordnlp/SHP    — Stanford Human Preferences (Reddit long-form QA)
"""

import random
from typing import Any

from datasets import load_dataset

from .config import CFG


# ── Individual loaders ────────────────────────────────────────────────────────

def load_hh_rlhf() -> list[dict[str, str]]:
    """Load and pre-process Anthropic/hh-rlhf preference pairs."""
    print("  Loading Anthropic/hh-rlhf ...")
    raw = load_dataset("Anthropic/hh-rlhf", split="train", streaming=False)

    def extract(ex: dict[str, Any]) -> dict[str, str]:
        return {"chosen": ex["chosen"][-300:], "rejected": ex["rejected"][-300:]}

    total = CFG["num_agents"] * CFG["n_samples_agent"] * 3
    pairs = raw.select(range(min(total, len(raw)))).map(
        extract, remove_columns=raw.column_names
    )
    return [
        p for p in pairs
        if len(p["chosen"]) > 20
        and len(p["rejected"]) > 20
        and p["chosen"] != p["rejected"]
    ]


def load_shp() -> list[dict[str, str]]:
    """Load and pre-process stanfordnlp/SHP preference pairs."""
    print("  Loading stanfordnlp/SHP ...")
    raw = load_dataset("stanfordnlp/SHP", split="train", streaming=False)

    pairs: list[dict[str, str]] = []
    target = CFG["num_agents"] * CFG["n_samples_agent"] * 3

    for ex in raw:
        if ex["score_A"] == ex["score_B"]:
            continue
        if ex["score_A"] > ex["score_B"]:
            chosen, rejected = ex["human_ref_A"], ex["human_ref_B"]
        else:
            chosen, rejected = ex["human_ref_B"], ex["human_ref_A"]

        chosen   = chosen[-300:]
        rejected = rejected[-300:]

        if len(chosen) > 20 and len(rejected) > 20 and chosen != rejected:
            pairs.append({"chosen": chosen, "rejected": rejected})

        if len(pairs) >= target:
            break

    return pairs


# ── Registry ──────────────────────────────────────────────────────────────────

DATASET_LOADERS: dict[str, Any] = {
    "hh-rlhf": load_hh_rlhf,
    "SHP":     load_shp,
}


# ── Partitioning ──────────────────────────────────────────────────────────────

def partition_to_agents(
    pairs: list[dict[str, str]],
) -> dict[int, list[dict[str, str]]]:
    """Shuffle and split pairs into disjoint per-agent slices."""
    random.shuffle(pairs)
    N, n = CFG["num_agents"], CFG["n_samples_agent"]
    agent_data = {i: pairs[i * n:(i + 1) * n] for i in range(N)}
    print(
        f"    Usable: {len(pairs)}, "
        f"per agent: {[len(agent_data[i]) for i in range(N)]}"
    )
    return agent_data
