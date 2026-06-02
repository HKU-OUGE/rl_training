"""Plots for the M20 obstacle-course eval output.

Reuses style / load_run / _save / short_terrain_label from plot_moe_eval_v2.
Produces 4 figures per call:
    course_01_progress_cdf.{pdf,png,svg}
    course_02_patch_pass.{pdf,png,svg}
    course_03_first_fail_hist.{pdf,png,svg}
    course_04_routing_by_patch.{pdf,png,svg}

When given multiple --data_dir entries, the progress CDF overlays the variants
(one line per variant). With a single --data_dir, it produces a single-line CDF.

Usage:
    python plot_course.py --data_dir <run_dir>
    python plot_course.py --data_dir <full_run> <A1_run> <B2_run> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pull the global style block + helpers from plot_moe_eval_v2.
sys.path.append(str(Path(__file__).resolve().parent))
from plot_moe_eval_v2 import (  # noqa: E402
    load_run,
    _save,
    short_terrain_label,
    ACCENT,
    CMAP_SEQ,
)


def compute_progress(raw, summary):
    """progress_ratio = clip(max_x_reached / course_length, 0, 1).

    max_x_reached is the per-env max disp_x reached. course_length from summary
    (32.0 m for the canonical v2 12-patch course; 45.0 m for the v1 6-patch).
    """
    course_length = float(summary.get("course_length", 32.0))
    max_x = np.asarray(raw["max_x_reached"], dtype=np.float32)
    progress = np.clip(max_x / course_length, 0.0, 1.0)
    return progress


def plot_progress_cdf(runs, out_dir, label_key="ablation"):
    """Per-variant CDF of progress_ratio.

    runs: list of (label, raw, summary) tuples.
    """
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for label, raw, summary in runs:
        p = compute_progress(raw, summary)
        p_sorted = np.sort(p)
        cdf = np.arange(1, len(p_sorted) + 1) / len(p_sorted)
        ax.plot(p_sorted, cdf, lw=1.4, label=str(label))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Progress ratio")
    ax.set_ylabel("Cumulative fraction of envs")
    ax.set_title("Course progress CDF")
    ax.grid(linestyle=":", linewidth=0.4, color="#cccccc", alpha=0.8)
    ax.set_axisbelow(True)
    if len(runs) > 1:
        ax.legend(loc="lower right", fontsize=6, frameon=False)
    out = out_dir / "course_01_progress_cdf.pdf"
    _save(fig, out)
    return out


def plot_patch_pass_bars(runs, summary_ref, out_dir):
    """Bar chart: pass-rate per patch, grouped by variant.

    runs: list of (label, raw, summary) tuples.
    """
    sub_names = summary_ref["sub_terrain_names"]
    n_patches = len(sub_names)
    width = 0.8 / max(1, len(runs))

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(n_patches)
    for i, (label, raw, summary) in enumerate(runs):
        pp = np.asarray(raw["per_patch_pass"], dtype=bool)  # (E, P)
        rates = pp.mean(axis=0)
        offset = (i - (len(runs) - 1) / 2) * width
        ax.bar(x + offset, rates, width=width, label=str(label), edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels([short_terrain_label(n) for n in sub_names],
                       rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-patch pass rate")
    ax.set_title("Patch completion across variants")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", alpha=0.8)
    ax.set_axisbelow(True)
    if len(runs) > 1:
        ax.legend(loc="upper right", fontsize=6, frameon=False)
    out = out_dir / "course_02_patch_pass.pdf"
    _save(fig, out)
    return out


def plot_first_fail_histogram(runs, summary_ref, out_dir):
    """Histogram: which patch did envs first fail at (failed envs only)."""
    sub_names = summary_ref["sub_terrain_names"]
    n_patches = len(sub_names)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    x = np.arange(n_patches)
    width = 0.8 / max(1, len(runs))
    for i, (label, raw, summary) in enumerate(runs):
        ff = np.asarray(raw["first_fail_patch"], dtype=np.int32)
        # ff == -1 means env reached the goal; ignore those for the histogram
        # (we want WHERE envs failed, not how many succeeded).
        ff_failed = ff[ff >= 0]
        if ff_failed.size == 0:
            counts = np.zeros(n_patches, dtype=np.float32)
        else:
            counts, _ = np.histogram(ff_failed, bins=np.arange(n_patches + 1))
        offset = (i - (len(runs) - 1) / 2) * width
        ax.bar(x + offset, counts, width=width, label=str(label), edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([short_terrain_label(n) for n in sub_names],
                       rotation=30, ha="right")
    ax.set_ylabel("Count of envs failing here first")
    ax.set_title("First-failure patch distribution")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", alpha=0.8)
    ax.set_axisbelow(True)
    if len(runs) > 1:
        ax.legend(loc="upper right", fontsize=6, frameon=False)
    out = out_dir / "course_03_first_fail_hist.pdf"
    _save(fig, out)
    return out


def plot_routing_by_patch(raw, summary, out_dir):
    """Mean gate weight binned by which patch the robot was on at each step.

    Heatmap rows = sub-terrains (6 patches); cols = leg experts.
    Per-step patch index is derived from disp_x: bucketed by the
    patch_end_x boundaries.
    """
    patch_end_x = np.asarray(summary["patch_end_x"], dtype=np.float32)  # (P,)
    sub_names = summary["sub_terrain_names"]
    n_patches = len(sub_names)

    gate_leg = np.asarray(raw["gate_leg"], dtype=np.float32)   # (E, T, K_leg)
    gate_wheel = np.asarray(raw["gate_wheel"], dtype=np.float32)  # (E, T, K_wh)
    pos = np.asarray(raw["root_pos_xy"], dtype=np.float32)     # (E, T, 2)
    term_step = np.asarray(raw["term_step"], dtype=np.int32)   # (E,)
    E, T, K_leg = gate_leg.shape
    K_wh = gate_wheel.shape[-1]

    # disp_x at each step = pos.x - pos.x[t=0]   (env spawn point)
    spawn_x = pos[:, 0, 0:1]  # (E, 1)
    disp_x = pos[..., 0] - spawn_x                              # (E, T)

    # Per-step active mask: t < term_step (or full T if env never terminated)
    last = np.where(term_step >= 0, term_step, T)
    t_idx = np.arange(T)[None, :]                               # (1, T)
    active = t_idx < last[:, None]                              # (E, T)

    # Patch index at each step: for disp_x, count how many boundaries we've crossed.
    # patch_idx = number of entries in patch_end_x where disp_x >= that entry,
    # capped at n_patches - 1 (anything past the last boundary stays as "patch n-1").
    # Vectorised:
    patch_idx = np.zeros_like(disp_x, dtype=np.int32)
    # patch_idx counts how many END-X thresholds are <= disp_x (this returns the
    # NEXT patch the robot is heading into). We want CURRENT patch, so subtract 1
    # for steps where any threshold passed... but the patch the robot is currently
    # ON is the FIRST one whose END-X > disp_x. Use np.searchsorted: it returns the
    # index `i` such that boundaries[i-1] <= x < boundaries[i].
    for i in range(E):
        patch_idx[i] = np.searchsorted(patch_end_x, disp_x[i], side="right")
    patch_idx = np.clip(patch_idx, 0, n_patches - 1)

    # Accumulate gate weights per (patch, expert)
    leg_sum = np.zeros((n_patches, K_leg), dtype=np.float64)
    leg_cnt = np.zeros(n_patches, dtype=np.int64)
    wh_sum = np.zeros((n_patches, K_wh), dtype=np.float64)
    wh_cnt = np.zeros(n_patches, dtype=np.int64)

    flat_active = active.reshape(-1)
    flat_patch = patch_idx.reshape(-1)
    flat_leg = gate_leg.reshape(-1, K_leg)
    flat_wh = gate_wheel.reshape(-1, K_wh)

    for p in range(n_patches):
        m = flat_active & (flat_patch == p)
        if m.any():
            leg_sum[p] = flat_leg[m].sum(axis=0)
            leg_cnt[p] = int(m.sum())
            wh_sum[p] = flat_wh[m].sum(axis=0)
            wh_cnt[p] = int(m.sum())
    leg_mat = leg_sum / np.maximum(leg_cnt[:, None], 1)
    wh_mat = wh_sum / np.maximum(wh_cnt[:, None], 1)

    fig, (ax_l, ax_w) = plt.subplots(
        1, 2, figsize=(6.4, 2.4),
        gridspec_kw={"width_ratios": [K_leg + 0.5, K_wh + 0.5], "wspace": 0.45},
    )
    for ax, mat, K, title in [
        (ax_l, leg_mat, K_leg, "Leg gate by patch"),
        (ax_w, wh_mat, K_wh, "Wheel gate by patch"),
    ]:
        im = ax.imshow(mat, cmap=CMAP_SEQ, vmin=0,
                       vmax=mat.max() if mat.size and not np.isnan(mat).all() else 1,
                       aspect="auto", origin="lower", interpolation="nearest")
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"E{i+1}" for i in range(K)])
        ax.set_yticks(range(n_patches))
        ax.set_yticklabels([short_terrain_label(n) for n in sub_names])
        ax.set_title(title)
        ax.tick_params(axis="both", which="both", bottom=False, left=False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    out = out_dir / "course_04_routing_by_patch.pdf"
    _save(fig, out)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Plot M20 obstacle-course eval outputs.")
    ap.add_argument("--data_dir", type=str, nargs="+", required=True,
                    help="One or more eval output dirs (each containing raw.npz + summary.json).")
    ap.add_argument("--label", type=str, nargs="*", default=None,
                    help="Optional per-variant labels (must match length of --data_dir).")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Override output dir for plots; default = first data_dir / plots.")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dirs = [Path(d) for d in args.data_dir]
    runs = []
    for d in data_dirs:
        raw, summary = load_run(d)
        runs.append((summary.get("ablation", d.name), raw, summary))

    if args.label is not None:
        assert len(args.label) == len(runs), "--label must match --data_dir length"
        runs = [(lbl, r, s) for lbl, (_, r, s) in zip(args.label, runs)]

    out_dir = Path(args.out_dir) if args.out_dir else (data_dirs[0] / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_progress_cdf(runs, out_dir)
    summary_ref = runs[0][2]
    plot_patch_pass_bars(runs, summary_ref, out_dir)
    plot_first_fail_histogram(runs, summary_ref, out_dir)
    # Routing-by-patch only uses the first variant (per-variant heatmaps would
    # multiply the figure count; downstream scripts can call this fn directly
    # if they want per-variant heatmaps).
    plot_routing_by_patch(runs[0][1], runs[0][2], out_dir)
    print(f"[plot_course] all plots written to {out_dir}")


if __name__ == "__main__":
    main()
