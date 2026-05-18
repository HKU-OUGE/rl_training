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


if __name__ == "__main__":
    main()
