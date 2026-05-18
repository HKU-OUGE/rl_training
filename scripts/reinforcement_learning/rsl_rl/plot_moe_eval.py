# scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py
"""Render 9 plots from eval_moe.py raw.npz output.

Usage: python plot_moe_eval.py --data_dir logs/moe_eval/<exp>/<run>
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(data_dir):
    raw = np.load(os.path.join(data_dir, "raw.npz"))
    with open(os.path.join(data_dir, "summary.json")) as f:
        summary = json.load(f)
    return raw, summary


def alive_mask(term_step, T):
    """Returns (N, T) bool. True iff env was in first episode at step t."""
    N = term_step.shape[0]
    t_idx = np.arange(T)[None, :]
    end = np.where(term_step < 0, T, term_step + 1)[:, None]
    return t_idx < end


def env_subterrain_name(types, summary):
    """Map per-env terrain_types (int col index) to sub-terrain string name."""
    col_map = summary["col_to_subterrain"]
    return np.array([col_map[int(t)] for t in types])


def aggregate_by_subterrain(values, subterrain_names, unique_names, agg="mean"):
    """Group `values` (N,) by sub-terrain name; return array of len(unique_names)."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--success_dist", type=float, default=None,
                   help="Override; default = summary.json's value")
    return p.parse_args()


def plot_success_heatmap(raw, summary, plots_dir):
    """Heatmap of success rate. Rows = level (0..num_rows-1), cols = unique sub-terrains."""
    term_cause = raw["term_cause"]
    levels = raw["terrain_levels"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]
    num_rows = summary["num_rows"]

    REACHED_GOAL = 4
    success = (term_cause == REACHED_GOAL)

    sub_per_env = env_subterrain_name(types, summary)

    M = np.full((num_rows, len(sub_names)), np.nan, dtype=np.float32)
    counts = np.zeros((num_rows, len(sub_names)), dtype=np.int32)
    for r in range(num_rows):
        for c, name in enumerate(sub_names):
            mask = (levels == r) & (sub_per_env == name)
            if mask.any():
                M[r, c] = success[mask].mean()
                counts[r, c] = mask.sum()

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(sub_names)), max(6, 0.2 * num_rows)))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(sub_names)))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels([f"L{r}" for r in range(num_rows)], fontsize=6)
    ax.set_xlabel("sub-terrain")
    ax.set_ylabel("difficulty level")
    ax.set_title(f"Success rate (reached +{summary['success_dist']:.1f}m within {summary['num_steps']*0.02:.0f}s)")
    plt.colorbar(im, ax=ax, label="success rate")

    # Annotate sample count in each cell
    for r in range(num_rows):
        for c in range(len(sub_names)):
            if counts[r, c] > 0:
                ax.text(c, r, str(counts[r, c]), ha="center", va="center", fontsize=5,
                        color="white" if M[r, c] < 0.5 else "black")

    plt.tight_layout()
    out = os.path.join(plots_dir, "01_success_heatmap.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")


def plot_expert_activation_bars(raw, summary, plots_dir):
    """Per sub-terrain stacked bar of avg leg/wheel expert weights."""
    gate_leg = raw["gate_leg"].astype(np.float32)    # (N, T, nL)
    gate_wheel = raw["gate_wheel"].astype(np.float32)  # (N, T, nW)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)  # (N, T)

    # mean over (env, time) where alive
    def avg_by_subterrain(gate):
        # gate: (N, T, K)
        am3 = am[:, :, None]
        per_env_avg = (gate * am3).sum(axis=1) / np.maximum(am.sum(axis=1, keepdims=True), 1)  # (N, K)
        sub_per_env = env_subterrain_name(types, summary)
        nK = gate.shape[-1]
        out = np.zeros((len(sub_names), nK))
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if mask.any():
                out[i] = per_env_avg[mask].mean(axis=0)
        return out

    leg_share = avg_by_subterrain(gate_leg)      # (13, nL)
    wheel_share = avg_by_subterrain(gate_wheel)  # (13, nW)

    fig, (ax_l, ax_w) = plt.subplots(2, 1, figsize=(max(8, 0.7 * len(sub_names)), 7), sharex=True)

    def stacked(ax, data, prefix, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        nK = data.shape[1]
        bottom = np.zeros(data.shape[0])
        for k in range(nK):
            ax.bar(range(data.shape[0]), data[:, k], bottom=bottom,
                   color=cmap(k / max(1, nK - 1)), label=f"{prefix}{k}", edgecolor="white", linewidth=0.5)
            bottom += data[:, k]
        ax.set_ylim(0, 1)
        ax.set_ylabel("avg gate weight")
        ax.legend(loc="upper right", fontsize=7, ncol=nK)

    stacked(ax_l, leg_share, "L", "tab10")
    ax_l.set_title("Leg expert activation per sub-terrain")
    stacked(ax_w, wheel_share, "W", "Set2")
    ax_w.set_title("Wheel expert activation per sub-terrain")
    ax_w.set_xticks(range(len(sub_names)))
    ax_w.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(plots_dir, "02_expert_activation_bars.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")


def plot_velocity_tracking_box(raw, summary, plots_dir):
    """3-subplot boxplot of |cmd - actual| for vx, vy, wz, grouped by sub-terrain."""
    cmd = raw["cmd"].astype(np.float32)            # (N, T, 3)
    actual = raw["actual_vel"].astype(np.float32)  # (N, T, 3)
    term_step = raw["term_step"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]

    T = cmd.shape[1]
    am = alive_mask(term_step, T)
    err = np.abs(cmd - actual)  # (N, T, 3)

    sub_per_env = env_subterrain_name(types, summary)

    # collect per-sub-terrain error samples (flatten over env+time where alive)
    fig, axes = plt.subplots(3, 1, figsize=(max(8, 0.7 * len(sub_names)), 9), sharex=True)
    labels = ["vx_err [m/s]", "vy_err [m/s]", "wz_err [rad/s]"]
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
        ax.set_ylabel(labels[ax_i])
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(range(1, len(sub_names) + 1))
    axes[-1].set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_title(f"Velocity tracking error per sub-terrain (cmd_vx={summary['cmd_vx']:.2f} m/s)")
    plt.tight_layout()
    out = os.path.join(plots_dir, "03_velocity_tracking_box.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")


def plot_termination_reward_breakdown(raw, summary, plots_dir):
    term_cause = raw["term_cause"]
    types = raw["terrain_types"]
    sub_names = summary["sub_terrain_names"]
    term_enum = {int(k): v for k, v in summary["term_enum"].items()}
    sub_per_env = env_subterrain_name(types, summary)

    # ---- top: stacked bar termination cause % per sub-terrain ----
    cause_names = ["time_out", "illegal_contact", "terrain_out_of_bounds", "bad_orientation", "reached_goal"]
    cause_enums = [0, 1, 2, 3, 4]
    cause_share = np.zeros((len(sub_names), len(cause_enums)))
    for i, name in enumerate(sub_names):
        mask = sub_per_env == name
        if not mask.any():
            continue
        for j, e in enumerate(cause_enums):
            cause_share[i, j] = (term_cause[mask] == e).mean()

    # ---- bottom: reward terms heatmap (sub-terrain × top-8 reward terms) ----
    reward_terms_arr = raw["reward_terms"].astype(np.float32) if "reward_terms" in raw.files else None
    rew_names = summary.get("reward_term_names", [])

    if reward_terms_arr is not None and len(rew_names) > 0:
        rew_per_sub = np.zeros((len(sub_names), len(rew_names)))
        for i, name in enumerate(sub_names):
            mask = sub_per_env == name
            if mask.any():
                rew_per_sub[i] = reward_terms_arr[mask].mean(axis=0)
        # pick top-8 by absolute magnitude
        top_idx = np.argsort(-np.abs(rew_per_sub).mean(axis=0))[:8]
        rew_top = rew_per_sub[:, top_idx]
        rew_top_names = [rew_names[i] for i in top_idx]
    else:
        rew_top = None
        rew_top_names = []

    n_axes = 2 if rew_top is not None else 1
    fig, axes = plt.subplots(n_axes, 1, figsize=(max(8, 0.7 * len(sub_names)), 4 + 3 * n_axes))
    if n_axes == 1:
        axes = [axes]

    # top
    ax_t = axes[0]
    colors = ["#666", "#d62728", "#9467bd", "#bcbd22", "#2ca02c"]
    bottom = np.zeros(len(sub_names))
    for j, (cname, col) in enumerate(zip(cause_names, colors)):
        ax_t.bar(range(len(sub_names)), cause_share[:, j], bottom=bottom,
                 label=cname, color=col, edgecolor="white", linewidth=0.5)
        bottom += cause_share[:, j]
    ax_t.set_ylim(0, 1)
    ax_t.set_ylabel("episode fraction")
    ax_t.set_title("Termination cause per sub-terrain")
    ax_t.legend(loc="upper right", fontsize=8, ncol=5)
    ax_t.set_xticks(range(len(sub_names)))
    ax_t.set_xticklabels(sub_names if n_axes == 1 else [""] * len(sub_names),
                         rotation=45, ha="right", fontsize=8)

    # bottom
    if rew_top is not None:
        ax_r = axes[1]
        im = ax_r.imshow(rew_top.T, cmap="RdBu_r", aspect="auto",
                         vmin=-np.abs(rew_top).max(), vmax=np.abs(rew_top).max())
        ax_r.set_yticks(range(len(rew_top_names)))
        ax_r.set_yticklabels(rew_top_names, fontsize=8)
        ax_r.set_xticks(range(len(sub_names)))
        ax_r.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
        ax_r.set_title("Top-8 reward terms (mean per env per sub-terrain)")
        plt.colorbar(im, ax=ax_r, label="reward (Episode_Reward sum)")

    plt.tight_layout()
    out = os.path.join(plots_dir, "04_termination_reward_breakdown.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"[plot] {out}")


def plot_leg_wheel_coactivation(raw, summary, plots_dir):
    """Joint probability matrix: P(leg_expert=i, wheel_expert=j), averaged over (env, time, alive)."""
    gate_leg = raw["gate_leg"].astype(np.float32)    # (N, T, nL)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    term_step = raw["term_step"]

    T = gate_leg.shape[1]
    am = alive_mask(term_step, T)
    am3 = am[:, :, None, None]

    # outer product per (env, t)
    co = gate_leg[:, :, :, None] * gate_wheel[:, :, None, :]  # (N, T, nL, nW)
    co_sum = (co * am3).sum(axis=(0, 1))
    norm = max(am.sum(), 1)
    co_avg = co_sum / norm

    nL, nW = co_avg.shape
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * nW + 2), max(5, 0.6 * nL + 2)))
    im = ax.imshow(co_avg, cmap="magma", aspect="auto")
    ax.set_xticks(range(nW)); ax.set_xticklabels([f"W{j}" for j in range(nW)])
    ax.set_yticks(range(nL)); ax.set_yticklabels([f"L{i}" for i in range(nL)])
    ax.set_xlabel("wheel expert"); ax.set_ylabel("leg expert")
    ax.set_title("Leg × Wheel expert co-activation (joint avg weight)")
    for i in range(nL):
        for j in range(nW):
            ax.text(j, i, f"{co_avg[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if co_avg[i, j] < co_avg.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = os.path.join(plots_dir, "05_leg_wheel_coactivation.png")
    plt.savefig(out, dpi=140); plt.close()
    print(f"[plot] {out}")


def main():
    args = parse_args()
    raw, summary = load_data(args.data_dir)
    print(f"[plot] loaded from {args.data_dir}")
    print(f"[plot] N={summary['num_envs']} T={summary['num_steps']} "
          f"success_dist={summary['success_dist']}m")
    sub_names = summary["sub_terrain_names"]
    print(f"[plot] {len(sub_names)} sub-terrains: {sub_names}")

    plots_dir = os.path.join(args.data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[plot] plots dir: {plots_dir}")

    # 9 plots implemented in Tasks 12-20
    plot_success_heatmap(raw, summary, plots_dir)
    plot_expert_activation_bars(raw, summary, plots_dir)
    plot_velocity_tracking_box(raw, summary, plots_dir)
    plot_termination_reward_breakdown(raw, summary, plots_dir)
    plot_leg_wheel_coactivation(raw, summary, plots_dir)


if __name__ == "__main__":
    main()
