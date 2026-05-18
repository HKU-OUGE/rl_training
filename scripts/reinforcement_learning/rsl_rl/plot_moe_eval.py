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


if __name__ == "__main__":
    main()
