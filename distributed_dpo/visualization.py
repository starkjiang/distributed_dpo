"""
distributed_dpo/visualization.py
==================================
All Matplotlib figure generation for the Distributed DPO experiments.

Figure inventory
----------------
Per-dataset (saved to  ddpo_results/<dataset>/):
  fig1_comparative       — gradient norm & DPO loss vs rounds (4 algorithms)
  fig2_local_steps       — gradient norm for E ∈ {1, 3, 6}
  fig3_participation     — gradient norm for S ∈ {1, 3, 5}
  fig4_topology          — graph drawings + DPO loss + consensus-error scatter
  fig5_staleness         — gradient norm for q_max ∈ {0, 2, 5}
  fig6_summary_dashboard — 2×3 overview of all experiments

Cross-dataset (saved to  ddpo_results/):
  fig_cross_dataset_comparison — 3×2 side-by-side metrics
  fig_cross_dataset_loss       — DPO loss per algorithm for both datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from .config import COLORS, GT_COLORS, DS_COLORS, PLOT_RCPARAMS
from .utils  import smooth, get, savefig
from .algorithms import run_fed_dpo

# Apply shared Matplotlib defaults
plt.rcParams.update(PLOT_RCPARAMS)


# =============================================================================
# Per-dataset figures
# =============================================================================

def plot_comparative(rc, labels, subdir):
    """Figure 1 — gradient-norm and DPO-loss convergence for all 4 algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for key, label in labels.items():
        ax.semilogy(
            smooth(get(rc[key], "grad_norm_sq")),
            label=label, color=COLORS[key], linewidth=2,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel(r"$E[\|\nabla L\|^2]$ (log)")
    ax.set_title("(a) Gradient Norm Convergence", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    for key, label in labels.items():
        ax.semilogy(
            smooth(get(rc[key], "loss")),
            label=label, color=COLORS[key], linewidth=2,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("DPO Loss")
    ax.set_title("(b) DPO Training Loss", fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, "fig1_comparative", subdir)
    plt.close(fig)


def plot_comparative_multiseed(all_seed_rc, labels, subdir):
    """Multi-seed comparative with mean ± std shading (addresses Q5)."""
    algo_keys = list(labels.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for panel, metric, ylabel in [
        (axes[0], "grad_norm_sq", r"$E[\|\nabla L\|^2]$ (log)"),
        (axes[1], "loss",         "DPO Loss"),
    ]:
        for key in algo_keys:
            # Collect across seeds
            curves = np.array([smooth(get(seed_rc[key], metric))
                                for seed_rc in all_seed_rc])
            mu  = curves.mean(axis=0)
            std = curves.std(axis=0)
            rounds = np.arange(len(mu))
            panel.semilogy(rounds, mu, label=labels[key],
                           color=COLORS[key], linewidth=2)
            panel.fill_between(rounds,
                               np.maximum(mu - std, 1e-10),
                               mu + std,
                               color=COLORS[key], alpha=0.18)
        panel.set_xlabel("Round")
        panel.set_ylabel(ylabel)
        panel.legend(fontsize=14)

    axes[0].set_title("(a) Gradient Norm — Mean ± Std", fontweight="bold")
    axes[1].set_title("(b) DPO Loss — Mean ± Std", fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig1b_comparative_multiseed", subdir)
    plt.close(fig)


def plot_local_steps(ra, subdir):
    """Figure 2 — gradient norm for local steps E ∈ {1, 3, 6}."""
    e_colors = {"1": "#3A86FF", "3": "#FF006E", "6": "#FB5607"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel (a): convergence curves
    ax = axes[0]
    for E_str, hist in ra["local_steps"].items():
        ax.semilogy(smooth(get(hist, "grad_norm_sq")),
                    label=f"E={E_str}", color=e_colors[E_str], linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel(r"$E[\|\nabla L\|^2]$")
    ax.set_title("(a) Convergence vs Local Steps E", fontweight="bold")
    ax.legend(title="Local Steps E")

    # Panel (b): bar chart of final stationary gap — shows the floor tradeoff
    ax = axes[1]
    e_vals = [int(k) for k in sorted(ra["local_steps"].keys(), key=int)]
    final  = [float(np.mean(get(ra["local_steps"][str(e)], "grad_norm_sq")[-5:])) for e in e_vals]
    bars   = ax.bar([str(e) for e in e_vals], final,
                    color=[e_colors[str(e)] for e in e_vals], edgecolor="white", width=0.5)
    for b, v in zip(bars, final):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f"{v:.4f}",
                ha="center", va="bottom", fontsize=14)
    ax.set_xlabel("Local Steps E")
    ax.set_ylabel("Final Stationary Gap")
    ax.set_title("(b) Final Gap vs E", fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig2_local_steps", subdir)
    plt.close(fig)


def plot_local_steps_extended(agent_data, subdir, beta, rounds_extended=100):
    """
    Run E ablation for more rounds to make the drift floor difference visible.
    This directly answers AmMw Q3 — the plateau separation between E=1 and E=6.
    """
    e_colors = {"1": "#3A86FF", "3": "#FF006E", "6": "#FB5607"}
    print(f"  [Extended E ablation, R={rounds_extended}] ...")
    ra_ext = {}
    for E_val in [1, 3, 6]:
        ra_ext[str(E_val)] = run_fed_dpo(agent_data, rounds_extended, E_val, beta)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for E_str, hist in ra_ext.items():
        ax.semilogy(smooth(get(hist, "grad_norm_sq")),
                    label=f"E={E_str}", color=e_colors[E_str], linewidth=2)
    ax.axvline(x=40, color="gray", linestyle=":", linewidth=1.2,
               label="R=40 (original)")
    ax.set_xlabel("Round"); ax.set_ylabel(r"$E[\|\nabla L\|^2]$")
    ax.set_title(f"(a) Extended Convergence (R={rounds_extended})\n"
                 "Floor separation becomes visible", fontweight="bold")
    ax.legend(title="Local Steps E", fontsize=14)

    ax = axes[1]
    e_vals = [1, 3, 6]
    final  = [float(np.mean(get(ra_ext[str(e)], "grad_norm_sq")[-5:])) for e in e_vals]
    bars   = ax.bar([str(e) for e in e_vals], final,
                    color=[e_colors[str(e)] for e in e_vals], edgecolor="white", width=0.5)
    for b, v in zip(bars, final):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f"{v:.5f}",
                ha="center", va="bottom", fontsize=14)
    ax.set_xlabel("Local Steps E"); ax.set_ylabel("Final Stationary Gap")
    ax.set_title(f"(b) Final Gap vs E at R={rounds_extended}\n"
                 r"(Theory: floor ∝ η²E²κ²)", fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig2b_local_steps_extended", subdir)
    plt.close(fig)
    return ra_ext


def plot_participation(ra, subdir):
    """Figure 3 — gradient norm for participation S ∈ {1, 3, 5}."""
    s_colors = {"1": "#E63946", "3": "#457B9D", "5": "#2DC653"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel (a): convergence curves
    ax = axes[0]
    for S_str, hist in ra["participation"].items():
        ax.semilogy(smooth(get(hist, "grad_norm_sq")),
                    label=f"S={S_str}", color=s_colors[S_str], linewidth=2)
    ax.set_xlabel("Round"); ax.set_ylabel(r"$E[\|\nabla L\|^2]$")
    ax.set_title("(a) Convergence under Partial Participation", fontweight="bold")
    ax.legend(title="Sampled Clients S")

    # Panel (b): scatter of final gap vs 1/S — directly shows ζ²g/S floor
    ax = axes[1]
    s_vals = [int(k) for k in sorted(ra["participation"].keys(), key=int)]
    final  = [float(np.mean(get(ra["participation"][str(s)], "grad_norm_sq")[-5:])) for s in s_vals]
    inv_s  = [1.0/s for s in s_vals]
    ax.scatter(inv_s, final, s=120, c=[s_colors[str(s)] for s in s_vals], zorder=5)
    for s, x, y in zip(s_vals, inv_s, final):
        ax.annotate(f"S={s}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=14)
    if len(inv_s) >= 2:
        z  = np.polyfit(inv_s, final, 1)
        xl = np.linspace(min(inv_s)*0.9, max(inv_s)*1.1, 50)
        ax.plot(xl, np.poly1d(z)(xl), "k--", alpha=0.6, linewidth=1.5, label="Linear fit")
        ax.legend(fontsize=14)
    ax.set_xlabel("1/S")
    ax.set_ylabel("Stationary Gap")
    ax.set_title("(b) Gap vs 1/S", fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig3_participation", subdir)
    plt.close(fig)


def plot_topology(ra, spectral_gaps, N, subdir):
    """
    Figure 4 — three panels:
      Left   : graph drawings (path / ring / star / complete)
      Center-left : DPO loss vs rounds per topology
      Center-right: final gradient norm per topology
      Right  : consensus error scatter vs spectral gap
    """
    gt_order = ["path", "ring", "star", "complete"]
    graph_nx = {
        "path":     nx.path_graph(N),
        "ring":     nx.cycle_graph(N),
        "star":     nx.star_graph(N - 1),
        "complete": nx.complete_graph(N),
    }

    fig = plt.figure(figsize=(22, 6))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.42)

    # Panel 1 — graph drawings
    ax_graphs = fig.add_subplot(gs[0, 0])
    ax_graphs.axis("off")
    ax_graphs.set_title("Communication Graphs", fontweight="bold", fontsize=10)
    inset_pos = [
        (0.02, 0.52, 0.46, 0.42), (0.52, 0.52, 0.46, 0.42),
        (0.02, 0.04, 0.46, 0.42), (0.52, 0.04, 0.46, 0.42),
    ]
    for gt, ip in zip(gt_order, inset_pos):
        ax_in = ax_graphs.inset_axes(ip)
        nx.draw_networkx(
            graph_nx[gt],
            pos=nx.circular_layout(graph_nx[gt]),
            ax=ax_in,
            node_color=GT_COLORS[gt],
            node_size=300,
            font_color="white",
            font_size=8,
            edge_color="#555",
        )
        gap = spectral_gaps[gt]["spectral_gap"]
        ax_in.set_title(f"{gt.capitalize()}\n1−ρ²={gap:.3f}", fontsize=9, fontweight="bold")
        ax_in.axis("off")

    # Panel 2 — DPO loss vs rounds
    ax_loss = fig.add_subplot(gs[0, 1])
    for gt in gt_order:
        hist = ra["graph_topology"][gt]
        gap  = spectral_gaps[gt]["spectral_gap"]
        ax_loss.plot(
            smooth(get(hist, "loss"), w=3),
            label=f"{gt.capitalize()} (1−ρ²={gap:.3f})",
            color=GT_COLORS[gt], linewidth=2.2,
        )
    ax_loss.set_xlabel("Communication Round")
    ax_loss.set_ylabel("DPO Loss (convergence error)")
    ax_loss.set_title(
        "(a) Convergence Error vs Topology\n(lower loss = better alignment)",
        fontweight="bold", fontsize=12,
    )
    ax_loss.legend(fontsize=12)

    # Panel 3: gradient norm vs rounds per topology — directly tests Theorem 6.1 ──
    ax_gn = fig.add_subplot(gs[0, 2])
    for gt in gt_order:
        hist = ra["graph_topology"][gt]
        gap  = spectral_gaps[gt]["spectral_gap"]
        ax_gn.semilogy(smooth(get(hist, "grad_norm_sq")),
                       label=f"{gt.capitalize()} (1−ρ²={gap:.3f})",
                       color=GT_COLORS[gt], linewidth=2.2)
    ax_gn.set_xlabel("Communication Round")
    ax_gn.set_ylabel(r"$E[\|\nabla L\|^2]$ (log)")
    ax_gn.set_title("(b) Gradient Norm vs Topology\n",
                    fontweight="bold", fontsize=12)
    ax_gn.legend(fontsize=12)

    # Panel 4 — consensus error vs spectral gap
    ax_ce = fig.add_subplot(gs[0, 3])
    gaps   = [spectral_gaps[gt]["spectral_gap"] for gt in gt_order]
    fin_ce = [
        float(np.mean(get(ra["graph_topology"][gt], "consensus_after")[-5:]))
        for gt in gt_order
    ]
    ax_ce.scatter(
        gaps, fin_ce, s=220,
        c=[GT_COLORS[gt] for gt in gt_order],
        edgecolors="white", linewidths=1.5, zorder=5,
    )
    for gt, gap, ce in zip(gt_order, gaps, fin_ce):
        ax_ce.annotate(
            gt.capitalize(), (gap, ce),
            textcoords="offset points", xytext=(8, 4), fontsize=12,
        )
    if min(gaps) > 0:
        gap_arr = np.linspace(min(gaps) * 0.7, max(gaps) * 1.15, 100)
        c_fit   = fin_ce[0] * gaps[0]
        ax_ce.plot(
            gap_arr, c_fit / gap_arr, "k--",
            alpha=0.6, linewidth=1.8,
            label=r"Theory: $\propto 1/(1-\rho^2)$",
        )
        ax_ce.legend(fontsize=12)
    ax_ce.set_xlabel("Spectral Gap  $1-\\rho^2$")
    ax_ce.set_ylabel("Final Consensus Error $e_\\theta$")
    ax_ce.set_title(
        "(b) Consensus Error vs Spectral Gap\n"
        r"(Theorem 6.1: $e_\theta \propto \eta^2/({1-\rho^2})$)",
        fontweight="bold", fontsize=12,
    )
    savefig(fig, "fig4_topology", subdir)
    plt.close(fig)


def plot_staleness(ra, subdir):
    """Figure 5 — gradient norm for staleness q_max ∈ {0, 2, 5}."""
    stale_colors = {"0": "#2DC653", "2": "#E9C46A", "5": "#E76F51"}
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for q, hist in ra["staleness"].items():
        ax.semilogy(
            smooth(get(hist, "grad_norm_sq")),
            label=f"$q_{{max}}={q}$", color=stale_colors[q], linewidth=2,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel(r"$E[\|\nabla L\|^2]$")
    ax.set_title("(a) Convergence under Staleness", fontweight="bold")
    ax.legend(title="Max Staleness")
    plt.tight_layout()
    savefig(fig, "fig5_staleness", subdir)
    plt.close(fig)


def plot_summary_dashboard(rc, ra, spectral_gaps, labels, R, ds_name, subdir):
    """Figure 6 — 2×3 summary dashboard."""
    gt_order     = ["path", "ring", "star", "complete"]
    gaps         = [spectral_gaps[gt]["spectral_gap"] for gt in gt_order]
    fin_ce       = [
        float(np.mean(get(ra["graph_topology"][gt], "consensus_after")[-5:]))
        for gt in gt_order
    ]
    e_colors     = {"1": "#3A86FF", "3": "#FF006E", "6": "#FB5607"}
    s_colors     = {"1": "#E63946", "3": "#457B9D", "5": "#2DC653"}
    stale_colors = {"0": "#2DC653", "2": "#E9C46A", "5": "#E76F51"}

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        f"Distributed DPO — Summary (5 agents, R={R})  [{ds_name}]",
        fontsize=14, fontweight="bold",
    )

    # [0,0] — comparative grad norm
    ax = axes[0, 0]
    for key, label in labels.items():
        ax.plot(smooth(get(rc[key], "grad_norm_sq")),
                label=label, color=COLORS[key], linewidth=2)
    ax.set_title("Comparative: Grad Norm", fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel(r"$\|\nabla L\|^2$"); ax.legend(fontsize=7)

    # [0,1] — local steps ablation
    ax = axes[0, 1]
    for E_str, hist in ra["local_steps"].items():
        ax.semilogy(smooth(get(hist, "grad_norm_sq")),
                    label=f"E={E_str}", color=e_colors[E_str], linewidth=2)
    ax.set_title("Ablation: Local Steps E", fontweight="bold")
    ax.set_xlabel("Round"); ax.legend(fontsize=8)

    # [0,2] — participation ablation
    ax = axes[0, 2]
    for S_str, hist in ra["participation"].items():
        ax.semilogy(smooth(get(hist, "grad_norm_sq")),
                    label=f"S={S_str}", color=s_colors[S_str], linewidth=2)
    ax.set_title("Ablation: Participation S", fontweight="bold")
    ax.set_xlabel("Round"); ax.legend(fontsize=8)

    # [1,0] — topology DPO loss
    ax = axes[1, 0]
    for gt in gt_order:
        hist = ra["graph_topology"][gt]
        gap  = spectral_gaps[gt]["spectral_gap"]
        ax.semilogy(smooth(get(hist, "loss"), w=3),
                    label=f"{gt} (1−ρ²={gap:.2f})",
                    color=GT_COLORS[gt], linewidth=2)
    ax.set_title("Topology: DPO Loss vs Rounds", fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("DPO Loss"); ax.legend(fontsize=7)

    # [1,1] — staleness ablation
    ax = axes[1, 1]
    for q, hist in ra["staleness"].items():
        ax.semilogy(smooth(get(hist, "grad_norm_sq")),
                    label=f"$q_{{max}}={q}$", color=stale_colors[q], linewidth=2)
    ax.set_title("Ablation: Staleness", fontweight="bold")
    ax.set_xlabel("Round"); ax.legend(fontsize=8)

    # [1,2] — spectral gap vs consensus error
    ax = axes[1, 2]
    ax.scatter(gaps, fin_ce, s=180,
               c=[GT_COLORS[gt] for gt in gt_order],
               edgecolors="white", linewidths=1.5, zorder=5)
    for gt, gap, ce in zip(gt_order, gaps, fin_ce):
        ax.annotate(gt.capitalize(), (gap, ce),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)
    if min(gaps) > 0:
        gap_arr = np.linspace(min(gaps) * 0.7, max(gaps) * 1.15, 100)
        c_fit   = fin_ce[0] * gaps[0]
        ax.semilogy(gap_arr, c_fit / gap_arr, "k--", alpha=0.5, linewidth=1.5)
    ax.set_title("Spectral Gap vs Consensus Error", fontweight="bold")
    ax.set_xlabel("1−ρ²"); ax.set_ylabel("Final $e_\\theta$")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "fig6_summary_dashboard", subdir)
    plt.close(fig)


# =============================================================================
# Cross-dataset comparison figures
# =============================================================================

def plot_cross_dataset(all_results, all_spectral_gaps):
    """
    3×2 cross-dataset comparison:
      Row 0 — comparative gradient norm
      Row 1 — topology DPO loss vs rounds
      Row 2 — spectral gap vs consensus error scatter
    """
    ds_names = list(all_results.keys())
    gt_order = ["path", "ring", "star", "complete"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        "Cross-Dataset Comparison: hh-rlhf vs SHP",
        fontsize=14, fontweight="bold",
    )

    for col, ds_name in enumerate(ds_names):
        rc, ra, spectral_gaps, labels, _ = all_results[ds_name]

        # Row 0 — comparative grad norm
        ax = axes[0, col]
        for key, label in labels.items():
            ax.semilogy(
                smooth(get(rc[key], "grad_norm_sq")),
                label=label, color=COLORS[key], linewidth=2,
            )
        ax.set_title(f"Comparative — Grad Norm\n[{ds_name}]", fontweight="bold")
        ax.set_xlabel("Round"); ax.set_ylabel(r"$\|\nabla L\|^2$"); ax.legend(fontsize=7)

        # Row 1 — topology DPO loss
        ax = axes[1, col]
        for gt in gt_order:
            hist = ra["graph_topology"][gt]
            gap  = spectral_gaps[gt]["spectral_gap"]
            ax.semilogy(
                smooth(get(hist, "loss"), w=3),
                label=f"{gt.capitalize()} (1−ρ²={gap:.3f})",
                color=GT_COLORS[gt], linewidth=2,
            )
        ax.set_title(f"Topology — DPO Loss vs Rounds\n[{ds_name}]", fontweight="bold")
        ax.set_xlabel("Round"); ax.set_ylabel("DPO Loss"); ax.legend(fontsize=7)

        # Row 2 — spectral gap vs consensus error
        ax = axes[2, col]
        gaps   = [spectral_gaps[gt]["spectral_gap"] for gt in gt_order]
        fin_ce = [
            float(np.mean(get(ra["graph_topology"][gt], "consensus_after")[-5:]))
            for gt in gt_order
        ]
        ax.scatter(
            gaps, fin_ce, s=200,
            c=[GT_COLORS[gt] for gt in gt_order],
            edgecolors="white", linewidths=1.5, zorder=5,
        )
        for gt, gap, ce in zip(gt_order, gaps, fin_ce):
            ax.annotate(gt.capitalize(), (gap, ce),
                        textcoords="offset points", xytext=(7, 4), fontsize=9)
        if min(gaps) > 0:
            gap_arr = np.linspace(min(gaps) * 0.7, max(gaps) * 1.15, 100)
            c_fit   = fin_ce[0] * gaps[0]
            ax.semilogy(gap_arr, c_fit / gap_arr, "k--", alpha=0.55, linewidth=1.8,
                        label=r"∝ $1/(1-\rho^2)$")
            ax.legend(fontsize=8)
        ax.set_title(f"Spectral Gap vs Consensus Error\n[{ds_name}]", fontweight="bold")
        ax.set_xlabel("Spectral Gap  $1-\\rho^2$")
        ax.set_ylabel("Final Consensus Error $e_\\theta$")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, "fig_cross_dataset_comparison")
    plt.close(fig)


def plot_cross_dataset_loss(all_results):
    """
    DPO training-loss overlay for both datasets, one subplot per algorithm.
    """
    ds_names   = list(all_results.keys())
    algo_keys  = ["centralized", "feddpo_full", "feddpo_partial", "decdpo_ring"]
    algo_titles = {
        "centralized":    "Centralized DPO",
        "feddpo_full":    "FedDPO (full)",
        "feddpo_partial": "FedDPO (partial)",
        "decdpo_ring":    "DecDPO (ring)",
    }
    line_styles = {"hh-rlhf": "-", "SHP": "--"}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    for ax, key in zip(axes, algo_keys):
        for ds_name in ds_names:
            rc = all_results[ds_name][0]
            ax.semilogy(
                smooth(get(rc[key], "loss")),
                label=ds_name,
                color=DS_COLORS[ds_name],
                linestyle=line_styles[ds_name],
                linewidth=2.2,
            )
        ax.set_title(algo_titles[key], fontweight="bold", fontsize=10)
        ax.set_xlabel("Communication Round")
        if ax is axes[0]:
            ax.set_ylabel("DPO Loss")
        ax.legend(fontsize=9)

    fig.suptitle(
        "DPO Loss Across Datasets: hh-rlhf vs SHP",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, "fig_cross_dataset_loss")
    plt.close(fig)
