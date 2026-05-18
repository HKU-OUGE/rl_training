# scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
"""Render 10 publication-quality plots from eval_moe.py raw.npz output.

Usage: python plot_moe_eval.py --data_dir logs/moe_eval/<exp>/<run>
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Style: SciencePlots → seaborn paper context → plain matplotlib (fallback)
# ---------------------------------------------------------------------------
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex", "grid"])
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False
    try:
        import seaborn as sns
        sns.set_theme(context="paper", style="whitegrid", palette="Set2")
    except ImportError:
        pass

mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
})

# ---------------------------------------------------------------------------
# Palette constants
# ---------------------------------------------------------------------------
SUBTERRAIN_CMAP = plt.get_cmap("tab20")
LEG_PALETTE = plt.get_cmap("Set2")
WHEEL_PALETTE = plt.get_cmap("Set3")
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"
HEATMAP_CMAP = "rocket_r"

try:
    mpl.colormaps["rocket_r"]
except Exception:
    HEATMAP_CMAP = "magma_r"

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_data(data_dir):
    raw = np.load(os.path.join(data_dir, "raw.npz"))
    with open(os.path.join(data_dir, "summary.json")) as f:
        summary = json.load(f)
    return raw, summary


def alive_mask(term_step, T):
    """Return (N, T) bool: True iff env was alive (first episode) at step t."""
    t_idx = np.arange(T)[None, :]
    end = np.where(term_step < 0, T, term_step + 1)[:, None]
    return t_idx < end


def env_subterrain_name(types, summary):
    """Map per-env terrain_types (int col index) to sub-terrain string name."""
    col_map = summary["col_to_subterrain"]
    return np.array([col_map[int(t)] for t in types])


def aggregate_by_subterrain(values, subterrain_names, unique_names, agg="mean"):
    """Group values (N,) by sub-terrain name; return array of len(unique_names)."""
    out = np.zeros(len(unique_names))
    for i, name in enumerate(unique_names):
        mask = subterrain_names == name
        if not mask.any():
            out[i] = np.nan
            continue
        if agg == "mean":
            out[i] = values[mask].mean()
        elif agg == "sum":
            out[i] = values[mask].sum()
        elif agg == "count":
            out[i] = mask.sum()
        else:
            raise ValueError(agg)
    return out


def add_title_strip(fig, summary):
    """Add small centered suptitle with run metadata."""
    iter_num = summary.get("iter", "?")
    N = summary.get("num_envs", "?")
    cmd_vx = summary.get("cmd_vx", "?")
    fig.suptitle(
        f"Rough-MoE-Teacher-Deeprobotics-M20-v0   ·   iter {iter_num}   ·   "
        f"N={N}   ·   cmd_vx={cmd_vx} m/s",
        fontsize=9, color="#555555", y=0.995, x=0.5,
    )


def compute_per_env_alive_means(gate, term_step, T):
    """Per-env mean of gate over alive steps.

    gate: (N, T, K) float
    term_step: (N,) int
    Returns (N, K) float
    """
    am = alive_mask(term_step, T)   # (N, T) bool
    am3 = am[:, :, None]
    summed = (gate.astype(np.float32) * am3).sum(axis=1)
    counts = np.maximum(am.sum(axis=1, keepdims=True), 1)
    return summed / counts


def _despine(ax):
    try:
        import seaborn as sns
        sns.despine(ax=ax)
    except ImportError:
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)


def _save(fig, out):
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def _subterrain_color(i, n):
    return SUBTERRAIN_CMAP(i / max(1, n - 1))


# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--success_dist", type=float, default=None,
                   help="Override; default = summary.json's value")
    return p.parse_args()


# ===========================================================================
# PLOT 00 – Dashboard (4-panel)
# ===========================================================================

def plot_dashboard(raw, summary, plots_dir):
    term_cause = raw["term_cause"]
    cmd = raw["cmd"].astype(np.float32)
    actual_vel = raw["actual_vel"].astype(np.float32)
    term_step = raw["term_step"]
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    sub_names = summary["sub_terrain_names"]
    T = summary["num_steps"]

    REACHED_GOAL = 4
    success_rate = (term_cause == REACHED_GOAL).mean()

    am = alive_mask(term_step, T)

    # Gate entropy
    eps = 1e-8
    def H_mean(gate, am):
        h = -(gate * np.log(gate + eps)).sum(axis=-1)  # (N, T)
        vals = h[am]
        return vals.mean() if vals.size > 0 else float("nan")

    leg_H = H_mean(gate_leg, am)
    wheel_H = H_mean(gate_wheel, am)
    nL = gate_leg.shape[-1]; nW = gate_wheel.shape[-1]
    max_leg_H = np.log(nL); max_wheel_H = np.log(nW)

    # cmd vs actual vx (alive samples)
    cmd_vx_flat = cmd[:, :, 0][am]
    actual_vx_flat = actual_vel[:, :, 0][am]

    # Termination cause fractions
    cause_labels = ["time_out", "illegal_contact", "terrain_oob", "bad_orient", "reached_goal"]
    cause_enums = [0, 1, 2, 3, 4]
    cause_colors = ["#aaaaaa", "#d62728", "#9467bd", "#bcbd22", "#2ca02c"]
    cause_fracs = np.array([(term_cause == e).mean() for e in cause_enums])

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # ── Top-left: big number success rate ──────────────────────────────────
    ax_sr = fig.add_subplot(gs[0, 0])
    ax_sr.axis("off")
    ax_sr.text(0.5, 0.62, f"{success_rate*100:.1f}%",
               ha="center", va="center", fontsize=56, fontweight="bold",
               color="#2ca02c" if success_rate > 0.5 else "#d62728",
               transform=ax_sr.transAxes)
    ax_sr.text(0.5, 0.22, "Overall Success Rate\n(reached goal = +4 m)",
               ha="center", va="center", fontsize=11, color="#444",
               transform=ax_sr.transAxes)
    ax_sr.set_title("Success Rate", fontsize=11)

    # ── Top-right: termination cause donut ─────────────────────────────────
    ax_dn = fig.add_subplot(gs[0, 1])
    non_zero = cause_fracs > 0
    wedges, texts, autotexts = ax_dn.pie(
        cause_fracs[non_zero],
        labels=[cause_labels[i] for i in range(len(cause_labels)) if non_zero[i]],
        colors=[cause_colors[i] for i in range(len(cause_colors)) if non_zero[i]],
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.5},
        startangle=90,
        textprops={"fontsize": 8},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax_dn.set_title("Termination Causes", fontsize=11)

    # ── Bottom-left: cmd vs actual vx hexbin ───────────────────────────────
    ax_hx = fig.add_subplot(gs[1, 0])
    hb = ax_hx.hexbin(cmd_vx_flat, actual_vx_flat,
                      gridsize=30, cmap=HEATMAP_CMAP, mincnt=1)
    plt.colorbar(hb, ax=ax_hx, label="count", pad=0.02)
    lim = max(abs(cmd_vx_flat).max(), abs(actual_vx_flat).max()) * 1.05
    ax_hx.plot([-lim, lim], [-lim, lim], "w--", lw=1, alpha=0.7, label="ideal")
    ax_hx.set_xlabel("cmd vx [m/s]")
    ax_hx.set_ylabel("actual vx [m/s]")
    ax_hx.set_title("cmd vs Actual vx (alive steps)", fontsize=11)
    ax_hx.legend(fontsize=8)
    _despine(ax_hx)

    # ── Bottom-right: gate entropy summary table ────────────────────────────
    ax_tb = fig.add_subplot(gs[1, 1])
    ax_tb.axis("off")
    table_data = [
        ["Metric", "Leg", "Wheel"],
        ["# experts (K)", f"{nL}", f"{nW}"],
        ["uniform H [nats]", f"{max_leg_H:.3f}", f"{max_wheel_H:.3f}"],
        ["mean H [nats]", f"{leg_H:.3f}", f"{wheel_H:.3f}"],
        ["H / H_uniform", f"{leg_H/max_leg_H:.3f}" if max_leg_H > 0 else "—",
                          f"{wheel_H/max_wheel_H:.3f}" if max_wheel_H > 0 else "—"],
    ]
    tbl = ax_tb.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    ax_tb.set_title("Gate Entropy Summary", fontsize=11)

    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "00_dashboard.png")
    _save(fig, out)


# ===========================================================================
# PLOT 01 – Success heatmap (30 levels × 12 sub-terrains)
# ===========================================================================

def plot_success_heatmap(raw, summary, plots_dir):
    term_cause = raw["term_cause"]
    levels = raw["terrain_levels"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]
    num_rows = summary["num_rows"]

    REACHED_GOAL = 4
    success = (term_cause == REACHED_GOAL)
    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)

    M = np.full((num_rows, n_sub), np.nan, dtype=np.float32)
    counts = np.zeros((num_rows, n_sub), dtype=np.int32)
    for r in range(num_rows):
        for c, name in enumerate(sub_names):
            mask = (levels == r) & (sub_per_env == name)
            if mask.any():
                M[r, c] = success[mask].mean()
                counts[r, c] = mask.sum()

    # Marginal row/col means (ignoring nan)
    row_mean = np.nanmean(M, axis=1)
    col_mean = np.nanmean(M, axis=0)

    fig_w = max(10, 0.7 * n_sub + 3)
    fig_h = max(7, 0.22 * num_rows + 1.5)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Leave space on right for row marginal strip and bottom for col strip
    ax = fig.add_axes([0.08, 0.18, 0.76, 0.72])
    ax_row = fig.add_axes([0.86, 0.18, 0.06, 0.72])  # right marginal
    ax_col = fig.add_axes([0.08, 0.06, 0.76, 0.10])  # bottom marginal

    im = ax.imshow(M, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1,
                   aspect="auto", origin="lower", interpolation="nearest")

    # Hatch missing cells (count == 0)
    for r in range(num_rows):
        for c in range(n_sub):
            if counts[r, c] == 0:
                patch = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   hatch="///", facecolor="#dddddd",
                                   edgecolor="#aaaaaa", linewidth=0)
                ax.add_patch(patch)
            elif counts[r, c] > 5 and not np.isnan(M[r, c]) and M[r, c] > 0:
                lum = M[r, c]
                tc = "white" if lum < 0.5 else "black"
                ax.text(c, r, f"{M[r,c]:.2f}", ha="center", va="center",
                        fontsize=5, color=tc)

    ax.set_xticks(range(n_sub))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(0, num_rows, 5))
    ax.set_yticklabels([f"L{r}" for r in range(0, num_rows, 5)], fontsize=6)
    ax.set_xlabel("sub-terrain")
    ax.set_ylabel("difficulty level")
    ax.set_title(
        f"Success Rate  (goal = +{summary['success_dist']:.1f} m  "
        f"within {summary['num_steps']*0.02:.0f} s)",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="success rate", shrink=0.8)

    # Row marginal
    ax_row.barh(range(num_rows), row_mean, color="#1f77b4", height=0.8, alpha=0.8)
    ax_row.set_xlim(0, 1); ax_row.set_ylim(-0.5, num_rows - 0.5)
    ax_row.set_yticks([]); ax_row.set_xlabel("mean", fontsize=7)
    ax_row.tick_params(labelsize=6)
    _despine(ax_row)

    # Col marginal
    ax_col.bar(range(n_sub), col_mean, color="#1f77b4", alpha=0.8)
    ax_col.set_xlim(-0.5, n_sub - 0.5)
    ax_col.set_ylim(0, 1); ax_col.set_xticks([]); ax_col.set_ylabel("mean", fontsize=7)
    ax_col.tick_params(labelsize=6)
    _despine(ax_col)

    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "01_success_heatmap.png")
    _save(fig, out)


# ===========================================================================
# PLOT 02 – Expert activation bars (stacked, 2×)
# ===========================================================================

def plot_expert_activation_bars(raw, summary, plots_dir):
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    leg_means = compute_per_env_alive_means(gate_leg, term_step, T)    # (N, nL)
    wheel_means = compute_per_env_alive_means(gate_wheel, term_step, T)  # (N, nW)

    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)
    nL = gate_leg.shape[-1]; nW = gate_wheel.shape[-1]

    def avg_by_sub(per_env, n_exp):
        out = np.zeros((n_sub, n_exp))
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if mask.any():
                out[i] = per_env[mask].mean(axis=0)
        return out

    leg_share = avg_by_sub(leg_means, nL)    # (n_sub, nL)
    wheel_share = avg_by_sub(wheel_means, nW)

    # Sort sub-terrains by leg-dominance (max(mean_weight) across leg experts)
    leg_dom_order = np.argsort(-leg_share.max(axis=1))
    leg_share_s = leg_share[leg_dom_order]
    wheel_share_s = wheel_share[leg_dom_order]
    sub_names_s = [sub_names[i] for i in leg_dom_order]

    fig, (ax_l, ax_w) = plt.subplots(2, 1,
                                      figsize=(max(10, 0.75 * n_sub), 8),
                                      sharex=True)

    def stacked_bars(ax, data, palette, prefix, n_exp):
        bottom = np.zeros(n_sub)
        for k in range(n_exp):
            col = palette(k / max(1, n_exp - 1))
            bars = ax.bar(range(n_sub), data[:, k], bottom=bottom,
                          color=col, label=f"{prefix}{k}",
                          edgecolor="white", linewidth=0.4)
            bottom += data[:, k]
        # Uniform baseline
        ax.axhline(1.0 / n_exp, ls="--", color="#555555", lw=1,
                   label=f"uniform (1/{n_exp})", alpha=0.7)
        # Per-expert annotation on right edge
        for k in range(n_exp):
            avg_k = data[:, k].mean()
            col = palette(k / max(1, n_exp - 1))
            ax.text(n_sub - 0.5 + 0.1 + k * 0.35, 0.5,
                    f"{prefix}{k}\n{avg_k:.2f}",
                    fontsize=6, color=col, va="center")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("avg gate weight")
        ax.legend(loc="upper left", fontsize=7, ncol=n_exp + 1)
        _despine(ax)

    stacked_bars(ax_l, leg_share_s, LEG_PALETTE, "L", nL)
    ax_l.set_title("Leg Expert Activation per Sub-terrain (sorted by dominance)", fontsize=11)

    stacked_bars(ax_w, wheel_share_s, WHEEL_PALETTE, "W", nW)
    ax_w.set_title("Wheel Expert Activation per Sub-terrain", fontsize=11)
    ax_w.set_xticks(range(n_sub))
    ax_w.set_xticklabels(sub_names_s, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "02_expert_activation_bars.png")
    _save(fig, out)


# ===========================================================================
# PLOT 03 – Velocity tracking violin (seaborn)
# ===========================================================================

def plot_velocity_tracking_violin(raw, summary, plots_dir):
    cmd = raw["cmd"].astype(np.float32)
    actual = raw["actual_vel"].astype(np.float32)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = cmd.shape[1]
    am = alive_mask(term_step, T)
    err = cmd - actual   # (N, T, 3); signed error

    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)

    try:
        import seaborn as sns
        import pandas as pd
        vel_labels = ["vx_err [m/s]", "vy_err [m/s]", "wz_err [rad/s]"]
        fig, axes = plt.subplots(3, 1,
                                  figsize=(max(10, 0.75 * n_sub), 10),
                                  sharex=True)
        for ax_i, ax in enumerate(axes):
            rows = []
            for name in sub_names:
                mask = sub_per_env == name
                if not mask.any():
                    continue
                vals = err[mask, :, ax_i][am[mask]]
                rows.extend([(v, name) for v in vals])
            if rows:
                df = pd.DataFrame(rows, columns=["error", "sub_terrain"])
                sns.violinplot(data=df, x="sub_terrain", y="error",
                               order=sub_names, ax=ax,
                               palette="mako", inner="quartile",
                               cut=0, linewidth=0.8)
            ax.axhline(0, ls="--", color="#555", lw=1, alpha=0.7)
            ax.set_ylabel(vel_labels[ax_i])
            ax.set_xlabel("")
            _despine(ax)
        axes[0].set_title("Velocity Tracking Error per Sub-terrain", fontsize=11)
        axes[-1].set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    except ImportError:
        # Fallback: boxplot
        fig, axes = plt.subplots(3, 1,
                                  figsize=(max(10, 0.75 * n_sub), 9),
                                  sharex=True)
        vel_labels = ["vx_err [m/s]", "vy_err [m/s]", "wz_err [rad/s]"]
        for ax_i, ax in enumerate(axes):
            data = []
            for name in sub_names:
                mask = sub_per_env == name
                if not mask.any():
                    data.append(np.array([]))
                    continue
                vals = err[mask, :, ax_i][am[mask]]
                data.append(vals)
            ax.boxplot(data, showfliers=False, widths=0.6)
            ax.axhline(0, ls="--", color="#555", lw=1, alpha=0.7)
            ax.set_ylabel(vel_labels[ax_i])
            _despine(ax)
        axes[0].set_title("Velocity Tracking Error per Sub-terrain", fontsize=11)
        axes[-1].set_xticks(range(1, n_sub + 1))
        axes[-1].set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "03_velocity_tracking_violin.png")
    _save(fig, out)


# ===========================================================================
# PLOT 04 – Routing heatmap (sub-terrain × expert) — MoE-Loco Fig 7 style
# ===========================================================================

def plot_routing_heatmap(raw, summary, plots_dir):
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    leg_means = compute_per_env_alive_means(gate_leg, term_step, T)    # (N, nL)
    wheel_means = compute_per_env_alive_means(gate_wheel, term_step, T)  # (N, nW)

    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)
    nL = gate_leg.shape[-1]; nW = gate_wheel.shape[-1]

    leg_mat = np.zeros((n_sub, nL))
    wheel_mat = np.zeros((n_sub, nW))
    for i, name in enumerate(sub_names):
        mask = sub_per_env == name
        if mask.any():
            leg_mat[i] = leg_means[mask].mean(axis=0)
            wheel_mat[i] = wheel_means[mask].mean(axis=0)

    fig, (ax_l, ax_w) = plt.subplots(1, 2,
                                      figsize=(max(10, 0.9 * (nL + nW) + 3), max(5, 0.55 * n_sub + 2)),
                                      gridspec_kw={"width_ratios": [nL, nW]})

    def draw_heatmap(ax, mat, col_labels, title, cmap):
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max() or 1,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(n_sub))
        ax.set_yticklabels(sub_names, fontsize=8)
        ax.set_xlabel("expert")
        ax.set_ylabel("sub-terrain")
        ax.set_title(title, fontsize=11)
        # Annotate every cell
        vmax = mat.max() if mat.max() > 0 else 1
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                tc = "white" if v < 0.5 * vmax else "black"
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color=tc)
        return im

    im_l = draw_heatmap(ax_l, leg_mat,
                        [f"L{k}" for k in range(nL)],
                        "Leg Routing Weights", HEATMAP_CMAP)
    im_w = draw_heatmap(ax_w, wheel_mat,
                        [f"W{k}" for k in range(nW)],
                        "Wheel Routing Weights", HEATMAP_CMAP)

    plt.colorbar(im_l, ax=ax_l, label="mean gate weight", shrink=0.7)
    plt.colorbar(im_w, ax=ax_w, label="mean gate weight", shrink=0.7)

    plt.tight_layout()
    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "04_routing_heatmap.png")
    _save(fig, out)


# ===========================================================================
# PLOT 05 – Leg × Wheel co-activation (with marginal bars)
# ===========================================================================

def plot_leg_wheel_coactivation(raw, summary, plots_dir):
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)
    am4 = am[:, :, None, None]

    co = gate_leg[:, :, :, None] * gate_wheel[:, :, None, :]  # (N, T, nL, nW)
    co_avg = (co * am4).sum(axis=(0, 1)) / max(am.sum(), 1)   # (nL, nW)

    nL, nW = co_avg.shape
    vmax = co_avg.max() if co_avg.max() > 0 else 1.0
    threshold = 0.5 * vmax

    fig = plt.figure(figsize=(max(6, 0.9 * nW + 3), max(5, 0.7 * nL + 3)))
    ax = fig.add_axes([0.12, 0.20, 0.65, 0.62])
    ax_top = fig.add_axes([0.12, 0.84, 0.65, 0.10])   # per-leg-expert total
    ax_right = fig.add_axes([0.79, 0.20, 0.08, 0.62])  # per-wheel-expert total

    im = ax.imshow(co_avg, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(nW)); ax.set_xticklabels([f"W{j}" for j in range(nW)], fontsize=9)
    ax.set_yticks(range(nL)); ax.set_yticklabels([f"L{i}" for i in range(nL)], fontsize=9)
    ax.set_xlabel("wheel expert"); ax.set_ylabel("leg expert")
    ax.set_title("Leg × Wheel Co-activation (joint avg weight)", fontsize=11)

    for i in range(nL):
        for j in range(nW):
            v = co_avg[i, j]
            if v > threshold * 0.2:
                tc = "white" if v < threshold else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color=tc)
    plt.colorbar(im, ax=ax, label="joint avg weight", shrink=0.8)

    # Marginal: per-leg-expert total (top bar)
    leg_totals = co_avg.sum(axis=1)
    ax_top.bar(range(nL), leg_totals, color=[LEG_PALETTE(k / max(1, nL - 1)) for k in range(nL)])
    ax_top.set_xlim(-0.5, nL - 0.5); ax_top.set_xticks([]); ax_top.set_ylabel("sum", fontsize=7)
    ax_top.set_title("Leg expert totals", fontsize=8)
    _despine(ax_top)

    # Marginal: per-wheel-expert total (right bar)
    wheel_totals = co_avg.sum(axis=0)
    ax_right.barh(range(nW), wheel_totals,
                  color=[WHEEL_PALETTE(k / max(1, nW - 1)) for k in range(nW)])
    ax_right.set_ylim(-0.5, nW - 0.5); ax_right.set_yticks(range(nW))
    ax_right.set_yticklabels([f"W{k}" for k in range(nW)], fontsize=7)
    ax_right.set_xlabel("sum", fontsize=7)
    ax_right.set_title("Wheel\ntotals", fontsize=8)
    _despine(ax_right)

    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "05_leg_wheel_coactivation.png")
    _save(fig, out)


# ===========================================================================
# PLOT 06 – Gate entropy (2×2: entropy bars + max-share bars)
# ===========================================================================

def plot_gate_entropy(raw, summary, plots_dir):
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)
    eps = 1e-8
    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)
    nL = gate_leg.shape[-1]; nW = gate_wheel.shape[-1]

    def entropy(g):
        return -(g * np.log(g + eps)).sum(axis=-1)  # (N, T)

    H_leg = entropy(gate_leg)
    H_wheel = entropy(gate_wheel)

    def avg_by_sub_am(H):
        out = np.full(n_sub, np.nan)
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if not mask.any():
                continue
            vals = H[mask][am[mask]]
            out[i] = vals.mean() if vals.size > 0 else np.nan
        return out

    leg_H = avg_by_sub_am(H_leg)
    wheel_H = avg_by_sub_am(H_wheel)

    # Max share (dominant expert's avg weight)
    leg_env_means = compute_per_env_alive_means(gate_leg, term_step, T)    # (N, nL)
    wheel_env_means = compute_per_env_alive_means(gate_wheel, term_step, T)

    def max_share_by_sub(means):
        out = np.full(n_sub, np.nan)
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if mask.any():
                out[i] = means[mask].max(axis=1).mean()
        return out

    leg_ms = max_share_by_sub(leg_env_means)
    wheel_ms = max_share_by_sub(wheel_env_means)

    max_leg_H = np.log(nL); max_wheel_H = np.log(nW)
    x = np.arange(n_sub)

    fig, axes = plt.subplots(2, 2, figsize=(max(14, 1.0 * n_sub), 8), sharex="col")
    ((ax_lh, ax_wh), (ax_lm, ax_wm)) = axes

    bar_kw = dict(edgecolor="white", linewidth=0.5)

    ax_lh.bar(x, leg_H, color=LEG_PALETTE(0.3), **bar_kw)
    ax_lh.axhline(max_leg_H, ls="--", color="#d62728", lw=1.2,
                  label=f"uniform H = log({nL}) = {max_leg_H:.2f}")
    ax_lh.axhline(0, ls=":", color="#555", lw=0.8, label="decisive H = 0")
    ax_lh.set_ylabel("entropy [nats]"); ax_lh.set_title("Leg Gate Entropy", fontsize=11)
    ax_lh.legend(fontsize=7); _despine(ax_lh)

    ax_wh.bar(x, wheel_H, color=WHEEL_PALETTE(0.3), **bar_kw)
    ax_wh.axhline(max_wheel_H, ls="--", color="#d62728", lw=1.2,
                  label=f"uniform H = log({nW}) = {max_wheel_H:.2f}")
    ax_wh.axhline(0, ls=":", color="#555", lw=0.8, label="decisive H = 0")
    ax_wh.set_ylabel("entropy [nats]"); ax_wh.set_title("Wheel Gate Entropy", fontsize=11)
    ax_wh.legend(fontsize=7); _despine(ax_wh)

    ax_lm.bar(x, leg_ms, color=LEG_PALETTE(0.6), **bar_kw)
    ax_lm.axhline(1.0 / nL, ls="--", color="#d62728", lw=1.2,
                  label=f"uniform = 1/{nL} = {1/nL:.3f}")
    ax_lm.axhline(1.0, ls=":", color="#555", lw=0.8, label="decisive = 1")
    ax_lm.set_ylabel("max gate share"); ax_lm.set_title("Leg Max Expert Share", fontsize=11)
    ax_lm.legend(fontsize=7); _despine(ax_lm)
    ax_lm.set_xticks(x); ax_lm.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)

    ax_wm.bar(x, wheel_ms, color=WHEEL_PALETTE(0.6), **bar_kw)
    ax_wm.axhline(1.0 / nW, ls="--", color="#d62728", lw=1.2,
                  label=f"uniform = 1/{nW} = {1/nW:.3f}")
    ax_wm.axhline(1.0, ls=":", color="#555", lw=0.8, label="decisive = 1")
    ax_wm.set_ylabel("max gate share"); ax_wm.set_title("Wheel Max Expert Share", fontsize=11)
    ax_wm.legend(fontsize=7); _despine(ax_wm)
    ax_wm.set_xticks(x); ax_wm.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "06_gate_entropy.png")
    _save(fig, out)


# ===========================================================================
# PLOT 07 – Expert switching & dominance (stacked bar + heatmap)
# ===========================================================================

def plot_expert_switching_dominance(raw, summary, plots_dir):
    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)
    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)
    nL = gate_leg.shape[-1]; nW = gate_wheel.shape[-1]

    dom_leg = gate_leg.argmax(axis=-1)    # (N, T)
    dom_wheel = gate_wheel.argmax(axis=-1)

    def switch_count(dom):
        diff = (dom[:, 1:] != dom[:, :-1]) & am[:, 1:] & am[:, :-1]
        return diff.sum(axis=1).astype(np.float32)

    sw_leg = switch_count(dom_leg)
    sw_wheel = switch_count(dom_wheel)

    sw_leg_per = aggregate_by_subterrain(sw_leg, sub_per_env, sub_names)
    sw_wheel_per = aggregate_by_subterrain(sw_wheel, sub_per_env, sub_names)

    # Dominant expert frequency heatmap
    def dom_freq_mat(dom, n_exp):
        mat = np.zeros((n_sub, n_exp))
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if not mask.any():
                continue
            dom_sub = dom[mask][am[mask]]  # flatten alive steps
            for k in range(n_exp):
                mat[i, k] = (dom_sub == k).mean()
        return mat

    leg_freq = dom_freq_mat(dom_leg, nL)
    wheel_freq = dom_freq_mat(dom_wheel, nW)

    fig = plt.figure(figsize=(max(12, 0.9 * n_sub), 11))
    gs = fig.add_gridspec(3, 1, hspace=0.55, height_ratios=[1, 1, 1])

    # -- Top: switches/episode grouped bar
    ax_sw = fig.add_subplot(gs[0])
    x = np.arange(n_sub); w = 0.35
    ax_sw.bar(x - w / 2, sw_leg_per, w, label="leg switches",
              color=LEG_PALETTE(0.3), edgecolor="white")
    ax_sw.bar(x + w / 2, sw_wheel_per, w, label="wheel switches",
              color=WHEEL_PALETTE(0.5), edgecolor="white")
    ax_sw.set_xticks(x); ax_sw.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    ax_sw.set_ylabel("avg switches / episode")
    ax_sw.set_title("Dominant Expert Switching Frequency per Sub-terrain", fontsize=11)
    ax_sw.legend(fontsize=8); _despine(ax_sw)

    # -- Middle: leg dominant-expert frequency heatmap
    ax_lh = fig.add_subplot(gs[1])
    im_l = ax_lh.imshow(leg_freq.T, cmap=HEATMAP_CMAP, vmin=0, vmax=1,
                         aspect="auto", interpolation="nearest")
    ax_lh.set_xticks(range(n_sub)); ax_lh.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
    ax_lh.set_yticks(range(nL)); ax_lh.set_yticklabels([f"L{k}" for k in range(nL)], fontsize=8)
    ax_lh.set_ylabel("leg expert"); ax_lh.set_title("Leg Dominant-Expert Frequency", fontsize=11)
    for r in range(nL):
        for c in range(n_sub):
            v = leg_freq[c, r]
            if v > 0.02:
                tc = "white" if v < 0.5 else "black"
                ax_lh.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6, color=tc)
    plt.colorbar(im_l, ax=ax_lh, label="fraction of steps", shrink=0.8)

    # -- Bottom: wheel dominant-expert frequency heatmap
    ax_wh = fig.add_subplot(gs[2])
    im_w = ax_wh.imshow(wheel_freq.T, cmap=HEATMAP_CMAP, vmin=0, vmax=1,
                         aspect="auto", interpolation="nearest")
    ax_wh.set_xticks(range(n_sub)); ax_wh.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=7)
    ax_wh.set_yticks(range(nW)); ax_wh.set_yticklabels([f"W{k}" for k in range(nW)], fontsize=8)
    ax_wh.set_ylabel("wheel expert"); ax_wh.set_title("Wheel Dominant-Expert Frequency", fontsize=11)
    for r in range(nW):
        for c in range(n_sub):
            v = wheel_freq[c, r]
            if v > 0.02:
                tc = "white" if v < 0.5 else "black"
                ax_wh.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6, color=tc)
    plt.colorbar(im_w, ax=ax_wh, label="fraction of steps", shrink=0.8)

    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "07_expert_switching_dominance.png")
    _save(fig, out)


# ===========================================================================
# PLOT 08 – Gating t-SNE (per-env mean gate output, MoE-Loco Fig 8 style)
# ===========================================================================

def plot_gating_tsne(raw, summary, plots_dir):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[plot] sklearn not installed — skipping plot 08 (gating t-SNE)")
        return

    gate_leg = raw["gate_leg"].astype(np.float32)    # (N, T, nL)
    gate_wheel = raw["gate_wheel"].astype(np.float32)  # (N, T, nW)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    leg_means = compute_per_env_alive_means(gate_leg, term_step, T)    # (N, nL)
    wheel_means = compute_per_env_alive_means(gate_wheel, term_step, T)  # (N, nW)

    # Concatenate: gating output vector per env
    X = np.concatenate([leg_means, wheel_means], axis=1)  # (N, nL+nW)
    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)

    print(f"[plot] gating t-SNE input: {X.shape}")

    import sklearn
    tsne_kwargs = dict(
        n_components=2,
        perplexity=min(30, max(5, X.shape[0] // 10)),
        random_state=0,
        init="pca",
    )
    sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_version >= (1, 4):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000

    Y = TSNE(**tsne_kwargs).fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(sub_names):
        mask = sub_per_env == name
        if not mask.any():
            continue
        ax.scatter(Y[mask, 0], Y[mask, 1],
                   color=SUBTERRAIN_CMAP(i / max(1, n_sub - 1)),
                   label=name, alpha=0.7, s=25)
    ax.set_title("Gating Output t-SNE (mean gate vector per env)", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2")
    _despine(ax)

    plt.tight_layout()
    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "08_gating_tsne.png")
    _save(fig, out)


# ===========================================================================
# PLOT 09 – Progress & survival (2-panel: median +x displacement + IQR, survival)
# ===========================================================================

def plot_progress_survival(raw, summary, plots_dir):
    root_pos_xy = raw["root_pos_xy"].astype(np.float32)  # (N, T, 2)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]
    success_dist = summary["success_dist"]
    T = summary["num_steps"]

    am = alive_mask(term_step, T)  # (N, T)
    sub_per_env = env_subterrain_name(types, summary)
    n_sub = len(sub_names)

    # +x displacement: root_pos_xy[:, t, 0] - root_pos_xy[:, 0, 0]
    origin_x = root_pos_xy[:, 0, 0:1]  # (N, 1)
    disp_x = root_pos_xy[:, :, 0] - origin_x  # (N, T)

    dt = 0.02
    t_axis = np.arange(T) * dt

    fig, (ax_prog, ax_surv) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for i, name in enumerate(sub_names):
        mask = sub_per_env == name
        if not mask.any():
            continue
        col = SUBTERRAIN_CMAP(i / max(1, n_sub - 1))
        disp_sub = disp_x[mask].copy()  # (n_sub_envs, T)
        # Mask out post-termination steps by setting to nan
        am_sub = am[mask]
        disp_sub[~am_sub] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            q50 = np.nanmedian(disp_sub, axis=0)
            q25 = np.nanpercentile(disp_sub, 25, axis=0)
            q75 = np.nanpercentile(disp_sub, 75, axis=0)

        ax_prog.plot(t_axis, q50, color=col, lw=1.5, label=name)
        ax_prog.fill_between(t_axis, q25, q75, color=col, alpha=0.2)

        surv = am_sub.mean(axis=0)
        ax_surv.plot(t_axis, surv, color=col, lw=1.5, label=name)

    ax_prog.axhline(success_dist, ls="--", color="#333", lw=1.2, alpha=0.8,
                    label=f"goal = {success_dist:.1f} m")
    ax_prog.set_ylabel("+x displacement [m]")
    ax_prog.set_title("Median +x Displacement per Sub-terrain (IQR shaded)", fontsize=11)
    ax_prog.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
    _despine(ax_prog)

    ax_surv.set_xlabel("time [s]")
    ax_surv.set_ylabel("fraction alive")
    ax_surv.set_ylim(0, 1.05)
    ax_surv.set_title("Survival Curve per Sub-terrain", fontsize=11)
    ax_surv.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=1)
    _despine(ax_surv)

    plt.tight_layout()
    add_title_strip(fig, summary)
    out = os.path.join(plots_dir, "09_progress_survival.png")
    _save(fig, out)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    raw, summary = load_data(args.data_dir)
    if args.success_dist is not None:
        summary["success_dist"] = args.success_dist

    print(f"[plot] loaded from {args.data_dir}")
    print(f"[plot] N={summary['num_envs']}  T={summary['num_steps']}  "
          f"iter={summary.get('iter','?')}  success_dist={summary['success_dist']} m")
    sub_names = summary["sub_terrain_names"]
    print(f"[plot] {len(sub_names)} sub-terrains: {sub_names}")

    plots_dir = os.path.join(args.data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[plot] plots dir: {plots_dir}")

    plot_fns = [
        ("00_dashboard",                plot_dashboard),
        ("01_success_heatmap",          plot_success_heatmap),
        ("02_expert_activation_bars",   plot_expert_activation_bars),
        ("03_velocity_tracking_violin", plot_velocity_tracking_violin),
        ("04_routing_heatmap",          plot_routing_heatmap),
        ("05_leg_wheel_coactivation",   plot_leg_wheel_coactivation),
        ("06_gate_entropy",             plot_gate_entropy),
        ("07_expert_switching_dominance", plot_expert_switching_dominance),
        ("08_gating_tsne",              plot_gating_tsne),
        ("09_progress_survival",        plot_progress_survival),
    ]

    for name, fn in plot_fns:
        try:
            fn(raw, summary, plots_dir)
        except Exception as exc:
            print(f"[plot] WARNING: {name} failed — {type(exc).__name__}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
