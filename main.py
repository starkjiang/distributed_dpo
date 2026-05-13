"""
main.py
=======
Entry point for Distributed DPO experiments.

Usage
-----
    python main.py                        # run all experiments (default config)
    python main.py --rounds 20            # quick smoke-test with 20 rounds
    python main.py --datasets hh-rlhf     # single dataset only
    python main.py --output ./my_results  # custom output directory
"""

import argparse
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from distributed_dpo import (
    CFG, SEED, DEVICE, MULTI_SEEDS,
    DATASET_LOADERS,
    init_models,
    partition_to_agents,
    run_centralized_dpo,
    run_fed_dpo,
    run_dec_dpo,
    run_experiments,
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
    save_results_json,
    print_summary,
    list_output_files,
    ds_tag,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed DPO experiments")
    p.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_LOADERS.keys()),
        default=list(DATASET_LOADERS.keys()),
        help="Datasets to run (default: all)",
    )
    p.add_argument(
        "--rounds", type=int, default=None,
        help="Override comm_rounds (e.g. 5 for a quick smoke-test)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Override output directory",
    )
    p.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    p.add_argument(
        "--multi_seed", type=list, default=MULTI_SEEDS,
        help=f"Multple seeds for comparative study (default: {MULTI_SEEDS})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Apply CLI overrides ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.rounds is not None:
        CFG["comm_rounds"] = args.rounds
    if args.output is not None:
        CFG["output_dir"] = args.output

    os.makedirs(CFG["output_dir"], exist_ok=True)
    print(f"Device  : {DEVICE}")
    print(f"Output  : {CFG['output_dir']}")
    print(f"Rounds  : {CFG['comm_rounds']}")
    print(f"Datasets: {args.datasets}")

    # ── Load models ───────────────────────────────────────────────────────────
    init_models()

    all_results: dict = {}

    for ds_name in args.datasets:
        loader = DATASET_LOADERS[ds_name]
        print(f"\n{'#' * 60}")
        print(f"  DATASET: {ds_name}")
        print(f"{'#' * 60}")

        pairs      = loader()
        agent_data = partition_to_agents(pairs)
        subdir     = ds_tag(ds_name)

        rc, ra, spectral_gaps, labels, rho_ring = run_experiments(agent_data, ds_name)
        all_results[ds_name] = (rc, ra, spectral_gaps, labels, rho_ring)

        # ── Per-dataset figures ───────────────────────────────────────────
        print(f"\n  Plotting [{ds_name}] ...")
        N = CFG["num_agents"]
        R = CFG["comm_rounds"]
        # E = CFG["local_steps"]

        plot_comparative(rc, labels, subdir)
        plot_local_steps(ra, subdir)
        plot_participation(ra, subdir)
        plot_topology(ra, spectral_gaps, N, subdir)
        plot_staleness(ra, subdir)
        plot_summary_dashboard(rc, ra, spectral_gaps, labels, R, ds_name, subdir)

        # ── Extended E ablation (R=100) ──────────────────────────────────────────────
        print (f"\n Extended E ablation (R=100) [{ds_name}] ...")
        plot_local_steps_extended(agent_data, subdir, CFG["beta"], rounds_extended=100)

        # ── Multi-seed comparative — mean ± std bands ──────────────────────────────────────────────
        print(f"\n  Multi-seed comparative (seeds={MULTI_SEEDS}) [{ds_name}] ...")
        all_seed_rc = []
        for seed in MULTI_SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            random.shuffle(list(agent_data.keys()))   # reshuffle agent assignment
            _rc = {}
            _rc["centralized"]    = run_centralized_dpo(agent_data, CFG["comm_rounds"], CFG["local_steps"], CFG["beta"])
            _rc["feddpo_full"]    = run_fed_dpo(agent_data, CFG["comm_rounds"], CFG["local_steps"], CFG["beta"])
            _rc["feddpo_partial"] = run_fed_dpo(agent_data, CFG["comm_rounds"], CFG["local_steps"], CFG["beta"], participation=CFG["participation"])
            h_dec2, _, _ = run_dec_dpo(agent_data, CFG["comm_rounds"], CFG["beta"], graph_type="ring")
            _rc["decdpo_ring"] = h_dec2
            all_seed_rc.append(_rc)
        plot_comparative_multiseed(all_seed_rc, labels, CFG["num_agents"],
                                   CFG["local_steps"], ds_name, subdir)

        # ── Per-dataset JSON ──────────────────────────────────────────────
        save_results_json(rc, ra, spectral_gaps, ds_name, subdir)
        print_summary(rc, ra, spectral_gaps, labels, ds_name)

    # ── Cross-dataset figures (only if multiple datasets were run) ────────────
    if len(all_results) > 1:
        print("\n  Plotting cross-dataset comparisons ...")
        plot_cross_dataset(
            all_results,
            {ds: all_results[ds][2] for ds in all_results},
        )
        plot_cross_dataset_loss(all_results)

    list_output_files(CFG["output_dir"])


if __name__ == "__main__":
    main()
