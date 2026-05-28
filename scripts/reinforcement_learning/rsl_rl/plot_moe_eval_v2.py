#!/usr/bin/env python3
"""Publication-quality eval plots for SplitMoE (v2).

Fixes vs v1:
  - Success computed from root_pos_xy displacement, not term_cause (the env
    config in v1 never fired reached_goal, so term_cause==REACHED_GOAL was
    always False and all heatmaps showed 0).
  - Sanity filter: |displacement| > MAX_DISPL m flagged invalid (sim glitch).
  - IEEE-style: sans-serif, compact, single-column friendly.
  - Vector PDF + PNG preview side-by-side.
  - Only emits the 3 figures the paper actually consumes:
      paper_01_success_heatmap.{pdf,png}
      paper_02_routing_specialization.{pdf,png}
      paper_03_velocity_tracking.{pdf,png}

Usage:
    python plot_moe_eval_v2.py                       # latest run, default dirs
    python plot_moe_eval_v2.py --data_dir <run_dir>  # specific run
    python plot_moe_eval_v2.py --success_dist 4.0    # override threshold
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Style — IEEE-ish, sans-serif, compact
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # embed TrueType so editors can edit text
    "ps.fonttype": 42,
    "figure.dpi": 110,
})

CMAP_SEQ = "magma"          # sequential, dark→light
CMAP_DIV = "RdBu_r"         # diverging if needed
ACCENT = "#1f6feb"          # GitHub-blue accent
GREY_HATCH = "#dcdcdc"

MAX_DISPL = 50.0   # |dx|>50m flagged sim glitch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run(data_dir: Path):
    raw = np.load(data_dir / "raw.npz", allow_pickle=True)
    with open(data_dir / "summary.json") as f:
        summary = json.load(f)
    return raw, summary


def compute_success(raw, success_dist: float, tol: float = 0.05):
    """Return (success_mask, displacement, valid_mask).

    Success is direction-aware. For each env we infer the command direction
    from the FIRST nonzero cmd_vx (raw['cmd'][:, :, 0]) and require the
    *signed* displacement in that direction to reach success_dist - tol.

      cmd_vx > 0  : success = (final_x - initial_x) >=  (success_dist - tol)
      cmd_vx < 0  : success = (final_x - initial_x) <= -(success_dist - tol)
      cmd_vx == 0 : env counted invalid

    A 5cm tolerance is applied because the env's reached_goal termination
    clamps the robot at success_dist (so max |dx| is success_dist - epsilon).

    Sim glitches with |dx|>MAX_DISPL are flagged invalid.
    """
    pos = raw["root_pos_xy"]                   # (E, T, 2)
    cmd = raw["cmd"]                            # (E, T, 3) — vx,vy,wz
    term_step = raw["term_step"]                # (E,)  -1 if no termination
    E, T, _ = pos.shape

    # Per-env command direction = sign of the FIRST nonzero cmd_vx in the rollout
    cmd_vx = cmd[..., 0]
    first_nonzero = np.argmax(np.abs(cmd_vx) > 1e-3, axis=1)
    cmd_first = cmd_vx[np.arange(E), first_nonzero]
    direction = np.sign(cmd_first)

    last_idx = np.where(term_step >= 0, term_step, T - 1).clip(max=T - 1)
    displacement = pos[np.arange(E), last_idx, 0] - pos[:, 0, 0]

    valid = (np.abs(displacement) <= MAX_DISPL) & (direction != 0)
    signed_dx = displacement * direction      # along command direction
    success = (signed_dx >= success_dist - tol) & valid
    return success, displacement, valid


def env_subterrain_name(types, summary):
    """Map per-env terrain type index to subterrain name string."""
    names = np.array(summary["sub_terrain_names"])
    # types is (E,) of int indices into the terrain matrix; col_to_subterrain
    # maps generator column → name. types is already a column index per env
    # in the standard eval_moe.py layout.
    col_to_sub = summary["col_to_subterrain"]
    # col_to_subterrain may be a list[str] or dict[str,str]; normalise
    if isinstance(col_to_sub, list):
        mapping = np.array(col_to_sub)
    else:
        mapping = np.array([col_to_sub[str(i)] for i in range(len(col_to_sub))])
    return mapping[types]


def short_terrain_label(name: str) -> str:
    """Short, paper-friendly labels (avoid wide x-tick)."""
    table = {
        "pyramid_stairs": "upstairs",
        "pyramid_stairs_inv": "downstairs",
        "stepping_stones": "gap",
        "rail": "bar",
        "hurdle_pole": "baffle-p",
        "hurdle_board": "baffle-b",
        "hurdle_wall": "baffle",
        "pit": "pit",
        "boxes": "boxes",
        "random_rough": "rough",
        "hf_pyramid_slope": "upslope",
        "hf_pyramid_slope_inv": "downslope",
    }
    return table.get(name, name)


def _save(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    fig.savefig(out_pdf.with_suffix(".svg"), format="svg")
    fig.savefig(out_pdf.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    return out_pdf


# ---------------------------------------------------------------------------
# Plot 01 — Success-rate heatmap
# ---------------------------------------------------------------------------

def plot_success_heatmap(raw, summary, success_mask, valid_mask, out_dir):
    levels = raw["terrain_levels"]                # (E,)
    types = raw["terrain_types"]                   # (E,)
    sub_names = summary["sub_terrain_names"]
    sub_per_env = env_subterrain_name(types, summary)
    num_rows = int(summary["num_rows"])
    n_sub = len(sub_names)

    # Bin difficulty levels into 6 groups (30 → 6 rows reads better)
    n_bins = 6
    bin_edges = np.linspace(0, num_rows, n_bins + 1, dtype=int)
    bin_labels = [f"L{bin_edges[i]}–{bin_edges[i+1]-1}" for i in range(n_bins)]

    M = np.full((n_bins, n_sub), np.nan, dtype=np.float32)
    counts = np.zeros((n_bins, n_sub), dtype=np.int32)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        for c, name in enumerate(sub_names):
            mask = (levels >= lo) & (levels < hi) & (sub_per_env == name) & valid_mask
            if mask.any():
                M[b, c] = success_mask[mask].mean()
                counts[b, c] = int(mask.sum())

    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    vmin, vmax = 0.4, 1.0
    im = ax.imshow(M, cmap=CMAP_SEQ, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="lower", interpolation="nearest")

    # Hatch / annotate
    for r in range(n_bins):
        for c in range(n_sub):
            if counts[r, c] == 0:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       hatch="///", facecolor=GREY_HATCH,
                                       edgecolor="none", linewidth=0))
            elif not np.isnan(M[r, c]):
                v = M[r, c]
                # Normalised position in colormap: below 0.7 → light text
                norm_v = (v - vmin) / (vmax - vmin)
                tc = "white" if norm_v < 0.45 else "black"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color=tc)

    ax.set_xticks(range(n_sub))
    ax.set_xticklabels([short_terrain_label(n) for n in sub_names],
                       rotation=35, ha="right")
    ax.set_yticks(range(n_bins))
    ax.set_yticklabels(bin_labels)
    ax.set_xlabel("Sub-terrain")
    ax.set_ylabel("Difficulty bin")
    ax.tick_params(axis="x", which="both", bottom=False)
    ax.tick_params(axis="y", which="both", left=False)

    overall = float(success_mask[valid_mask].mean()) if valid_mask.any() else 0.0
    ax.set_title(f"Success rate  ($d_x\\geq{summary['success_dist']:.0f}$ m, "
                 f"overall {overall*100:.0f}%)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=6, width=0.4, length=2)

    out = out_dir / "paper_01_success_heatmap.pdf"
    _save(fig, out)
    return out, overall


# ---------------------------------------------------------------------------
# Plot 02 — Routing specialization (heatmap of expert weight by terrain)
# ---------------------------------------------------------------------------

def plot_routing_specialization(raw, summary, valid_mask, out_dir):
    gate_leg = raw["gate_leg"].astype(np.float32)     # (E, T, K_leg)
    gate_wheel = raw["gate_wheel"].astype(np.float32) # (E, T, K_wh)
    types = raw["terrain_types"]
    term_step = raw["term_step"]
    sub_names = summary["sub_terrain_names"]
    sub_per_env = env_subterrain_name(types, summary)

    K_leg = gate_leg.shape[-1]
    K_wh = gate_wheel.shape[-1]

    # Average gate weights up to termination, per env, then group by terrain
    E, T, _ = gate_leg.shape
    valid_step_mask = np.zeros((E, T), dtype=bool)
    for e in range(E):
        last = int(term_step[e]) if term_step[e] >= 0 else T
        valid_step_mask[e, :last] = True

    def _mean_per_env(gate):  # → (E, K)
        sums = (gate * valid_step_mask[..., None]).sum(axis=1)
        counts = valid_step_mask.sum(axis=1, keepdims=True).clip(min=1)
        return sums / counts

    leg_per_env = _mean_per_env(gate_leg)
    wh_per_env = _mean_per_env(gate_wheel)

    # Group by sub-terrain (rows = terrain, cols = expert)
    leg_mat = np.full((len(sub_names), K_leg), np.nan)
    wh_mat = np.full((len(sub_names), K_wh), np.nan)
    leg_entropy = np.full(len(sub_names), np.nan)
    wh_entropy = np.full(len(sub_names), np.nan)
    for i, name in enumerate(sub_names):
        m = (sub_per_env == name) & valid_mask
        if m.any():
            lm = leg_per_env[m].mean(axis=0)
            wm = wh_per_env[m].mean(axis=0)
            leg_mat[i] = lm
            wh_mat[i] = wm
            leg_entropy[i] = _entropy(lm) / np.log(K_leg)
            wh_entropy[i] = _entropy(wm) / np.log(K_wh)

    # 3-panel: leg-routing | wheel-routing | normalized entropy bars
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.5, 2.6),
        gridspec_kw={"width_ratios": [K_leg + 0.5, K_wh + 0.5, 5.0],
                     "wspace": 0.55},
    )

    for ax, mat, K, title in [
        (ax1, leg_mat, K_leg, "Leg gate weight"),
        (ax2, wh_mat, K_wh, "Wheel gate weight"),
    ]:
        im = ax.imshow(mat, cmap=CMAP_SEQ, vmin=0, vmax=mat.max() if not np.isnan(mat).all() else 1,
                       aspect="auto", origin="lower", interpolation="nearest")
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"E{i+1}" for i in range(K)])
        ax.set_yticks(range(len(sub_names)))
        ax.set_yticklabels([short_terrain_label(n) for n in sub_names])
        ax.set_title(title)
        ax.tick_params(axis="both", which="both", bottom=False, left=False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2)

    # Panel 3 — normalized entropy (0 = peaked, 1 = uniform)
    y = np.arange(len(sub_names))
    width = 0.4
    ax3.barh(y - width / 2, leg_entropy, height=width, color=ACCENT,
             label="leg", edgecolor="none")
    ax3.barh(y + width / 2, wh_entropy, height=width, color="#f78c2a",
             label="wheel", edgecolor="none")
    ax3.set_yticks(range(len(sub_names)))
    ax3.set_yticklabels([short_terrain_label(n) for n in sub_names])
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Normalized gate entropy")
    ax3.set_title("Specialization (lower = peaked)")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, handlelength=1.2,
               columnspacing=1.0, handletextpad=0.4)
    ax3.tick_params(axis="y", which="both", left=False)
    ax3.grid(axis="x", linestyle=":", linewidth=0.4, color="#cccccc",
             alpha=0.8, zorder=0)
    ax3.set_axisbelow(True)

    out = out_dir / "paper_02_routing_specialization.pdf"
    _save(fig, out)
    return out


def _entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum().clip(min=1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# Plot 04 — Gate-output t-SNE (per-env clustering by sub-terrain)
# ---------------------------------------------------------------------------

def plot_gate_tsne(raw, summary, valid_mask, out_dir):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[warn] sklearn not installed, skipping t-SNE plot")
        return None

    gate_leg = raw["gate_leg"].astype(np.float32)
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    types = raw["terrain_types"]
    term_step = raw["term_step"]
    sub_names = summary["sub_terrain_names"]
    sub_per_env = env_subterrain_name(types, summary)

    E, T, _ = gate_leg.shape
    step_mask = np.zeros((E, T), dtype=bool)
    for e in range(E):
        last = int(term_step[e]) if term_step[e] >= 0 else T
        step_mask[e, :last] = True

    def _mean(gate):
        sums = (gate * step_mask[..., None]).sum(axis=1)
        cnts = step_mask.sum(axis=1, keepdims=True).clip(min=1)
        return sums / cnts

    leg_means = _mean(gate_leg)
    wh_means = _mean(gate_wheel)
    X = np.concatenate([leg_means, wh_means], axis=1)  # (E, K_leg+K_wh)

    # Filter: kept terrains + valid envs only
    kept = np.isin(sub_per_env, sub_names) & valid_mask
    X = X[kept]
    labels = sub_per_env[kept]

    if X.shape[0] < 5:
        print(f"[warn] too few envs for t-SNE ({X.shape[0]}), skipping")
        return None

    tsne_kwargs = dict(
        n_components=2,
        perplexity=min(30, max(5, X.shape[0] // 10)),
        random_state=0,
        init="pca",
        learning_rate="auto",
        n_jobs=1,  # single thread → fully deterministic given fixed random_state
    )
    np.random.seed(0)
    try:
        Y = TSNE(**tsne_kwargs).fit_transform(X)
    except TypeError:  # older sklearn
        tsne_kwargs.pop("learning_rate", None)
        Y = TSNE(**tsne_kwargs).fit_transform(X)

    # Build a stable color map across kept terrains
    n_sub = len(sub_names)
    cmap = plt.cm.tab20

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, name in enumerate(sub_names):
        m = labels == name
        if not m.any():
            continue
        lbl = short_terrain_label(name)
        lbl = lbl[:1].upper() + lbl[1:]
        ax.scatter(Y[m, 0], Y[m, 1], s=12, alpha=0.8,
                   color=cmap(i / max(1, n_sub - 1)),
                   label=lbl, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=6, frameon=False, markerscale=1.6,
              handlelength=0.6, labelspacing=0.4, borderpad=0.2)

    out = out_dir / "paper_04_gate_tsne.pdf"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Plot 05 — Leg gate routing t-SNE  (wheel gate is covered by plot 06)
# ---------------------------------------------------------------------------

def _eta2(X, groups):
    """Fraction of total variance in X (N,D) explained by categorical groups.

    eta^2 = SS_between / SS_total. 0 = factor explains nothing, 1 = all.
    """
    gtot = X.mean(axis=0)
    ss_tot = float(((X - gtot) ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        m = groups == g
        ss_between += int(m.sum()) * float(((X[m].mean(axis=0) - gtot) ** 2).sum())
    return ss_between / ss_tot


def plot_leg_routing(raw, summary, valid_mask, out_dir, merge_directions=False,
                     dir_encoding="panels"):
    """Leg-gate routing t-SNE, colored by terrain.

    The wheel gate is covered separately by plot_wheel_violin — a ternary
    simplex of the 3-D wheel gate packs every env into a tiny central blob and
    reads as undifferentiated even though terrain explains ~80% of its variance.
    With bidirectional command data the leg t-SNE is split into forward /
    backward panels, since direction is the dominant routing axis and pooling
    it masks the terrain structure; eta^2 (fraction of gate-vector variance
    explained) is computed within each direction.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[warn] sklearn not installed, skipping leg-routing plot")
        return None

    gate_leg = raw["gate_leg"].astype(np.float32)
    cmd = raw["cmd"].astype(np.float32)
    types = raw["terrain_types"]
    term_step = raw["term_step"]
    sub_names = summary["sub_terrain_names"]
    sub_per_env = env_subterrain_name(types, summary)

    E, T, _ = gate_leg.shape
    step_mask = np.zeros((E, T), dtype=bool)
    for e in range(E):
        last = int(term_step[e]) if term_step[e] >= 0 else T
        step_mask[e, :max(last, 1)] = True

    def _mean(gate):
        sums = (gate * step_mask[..., None]).sum(axis=1)
        cnts = step_mask.sum(axis=1, keepdims=True).clip(min=1)
        return sums / cnts

    leg_means = _mean(gate_leg)
    vx = _mean(cmd)[:, 0]

    kept = np.isin(sub_per_env, sub_names) & valid_mask
    leg_means = leg_means[kept]
    labels = sub_per_env[kept]
    vx = vx[kept]

    if leg_means.shape[0] < 5:
        print(f"[warn] too few envs for leg-routing ({leg_means.shape[0]}), skipping")
        return None

    n_sub = len(sub_names)
    cmap = plt.cm.tab20

    def _disp(name):
        lbl = short_terrain_label(name)
        return lbl[:1].upper() + lbl[1:]

    def _tsne(X):
        np.random.seed(0)
        kw = dict(n_components=2,
                  perplexity=min(30, max(5, X.shape[0] // 10)),
                  random_state=0, init="pca", learning_rate="auto",
                  n_jobs=1)  # single thread → fully deterministic
        try:
            return TSNE(**kw).fit_transform(X)
        except TypeError:
            kw.pop("learning_rate", None)
            kw.pop("n_jobs", None)
            return TSNE(**kw).fit_transform(X)

    def _leg_panel(ax, X, lbls, title):
        Y = _tsne(X)
        for i, name in enumerate(sub_names):
            m = lbls == name
            if m.any():
                ax.scatter(Y[m, 0], Y[m, 1], s=10, alpha=0.8,
                           color=cmap(i / max(1, n_sub - 1)),
                           label=_disp(name), edgecolors="white", linewidths=0.3)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)

    bidir = bool((vx > 0).any() and (vx < 0).any())

    # dir_encoding takes precedence over merge_directions
    if dir_encoding in ("shape_split", "shape_side", "fill_split", "fill_overlay") and bidir:
        # Single panel, color=terrain, shape/fill=direction
        eta_leg = _eta2(leg_means, labels)
        eta_l_dir = _eta2(leg_means, (vx > 0).astype(int))
        print(f"[info] leg-routing ({dir_encoding}) eta^2: terrain={eta_leg:.2f}  direction={eta_l_dir:.2f}")
        Y = _tsne(leg_means)
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        fwd = vx > 0
        for i, name in enumerate(sub_names):
            color = cmap(i / max(1, n_sub - 1))
            m_t = labels == name
            if not m_t.any():
                continue
            m_fwd, m_bwd = m_t & fwd, m_t & ~fwd
            if dir_encoding in ("fill_split", "fill_overlay"):
                # filled circle for fwd, filled triangle for bwd, both with white
                # halo edges. White halos separate overlapping markers in dense
                # scatter; bwd as filled triangle is more visible than open shapes
                # while still being identifiable by shape.
                if m_fwd.any():
                    ax.scatter(Y[m_fwd, 0], Y[m_fwd, 1], s=14, alpha=0.85,
                               color=color, marker="o", label=_disp(name),
                               edgecolors="white", linewidths=0.35)
                if m_bwd.any():
                    ax.scatter(Y[m_bwd, 0], Y[m_bwd, 1], s=18, alpha=0.85,
                               color=color, marker="^", label=None,
                               edgecolors="white", linewidths=0.35)
            else:
                # shape_split / shape_side: filled circle for fwd, filled triangle for bwd
                # (both with white halo edge to separate overlapping markers)
                if m_fwd.any():
                    ax.scatter(Y[m_fwd, 0], Y[m_fwd, 1], s=14, alpha=0.85,
                               color=color, marker="o", label=_disp(name),
                               edgecolors="white", linewidths=0.35)
                if m_bwd.any():
                    ax.scatter(Y[m_bwd, 0], Y[m_bwd, 1], s=18, alpha=0.85,
                               color=color, marker="^", label=None,
                               edgecolors="white", linewidths=0.35)
        # add a separate legend entry pair (color-neutral) for direction
        from matplotlib.lines import Line2D
        if dir_encoding in ("fill_split", "fill_overlay"):
            dir_handles = [Line2D([0], [0], marker="o", color="0.3", linestyle="",
                                  markersize=5, label="forward (●)"),
                           Line2D([0], [0], marker="^", color="0.3", linestyle="",
                                  markersize=5, label="backward (▲)")]
        else:
            dir_handles = [Line2D([0], [0], marker="o", color="0.3", linestyle="",
                                  markersize=5, label="forward (●)"),
                           Line2D([0], [0], marker="^", color="0.3", linestyle="",
                                  markersize=5, label="backward (▲)")]
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title(f"terrain $\\eta^2$={eta_leg:.2f}  direction $\\eta^2$={eta_l_dir:.2f}",
                     fontsize=8)
        # Two legends placed INSIDE the axes (matplotlib finds the emptiest
        # corner via loc='best'). White-ish frame keeps them readable on top
        # of the scatter cloud.
        terr_handles, terr_labels = ax.get_legend_handles_labels()
        leg_terr = ax.legend(terr_handles, terr_labels, loc="best",
                             fontsize=6, frameon=True, framealpha=0.85,
                             edgecolor="0.7", facecolor="white",
                             markerscale=1.2, handlelength=0.6,
                             labelspacing=0.3, borderpad=0.4,
                             title="terrain", title_fontsize=6)
        ax.add_artist(leg_terr)
        ax.legend(handles=dir_handles, loc="lower right",
                  fontsize=6, frameon=True, framealpha=0.85,
                  edgecolor="0.7", facecolor="white",
                  handlelength=0.6, labelspacing=0.4, borderpad=0.4,
                  title="direction", title_fontsize=6)
        out = out_dir / "paper_05_leg_routing.pdf"
        _save(fig, out)
        return out

    if bidir and not merge_directions:
        fwd, bwd = vx > 0, vx < 0
        eta_lf = _eta2(leg_means[fwd], labels[fwd])
        eta_lb = _eta2(leg_means[bwd], labels[bwd])
        eta_l_dir = _eta2(leg_means, (vx > 0).astype(int))
        print(f"[info] leg-routing (bidir) eta^2  direction={eta_l_dir:.2f}  "
              f"terrain|fwd={eta_lf:.2f}  terrain|bwd={eta_lb:.2f}")
        fig, (axf, axb) = plt.subplots(1, 2, figsize=(7.4, 3.6))
        _leg_panel(axf, leg_means[fwd], labels[fwd],
                   f"Leg gate t-SNE — forward\nterrain $\\eta^2$={eta_lf:.2f}")
        _leg_panel(axb, leg_means[bwd], labels[bwd],
                   f"Leg gate t-SNE — backward\nterrain $\\eta^2$={eta_lb:.2f}")
        fig.suptitle(f"Leg gate routing  (direction $\\eta^2$={eta_l_dir:.2f})",
                     fontsize=9)
        legend_ax = axf
        rect = [0, 0.08, 1, 0.93]
    else:
        eta_leg = _eta2(leg_means, labels)
        tag = "bidir merged" if (bidir and merge_directions) else "1-dir"
        print(f"[info] leg-routing ({tag}) eta^2: terrain={eta_leg:.2f}")
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        title_dir = " (fwd + bwd merged)" if (bidir and merge_directions) else ""
        _leg_panel(ax, leg_means, labels,
                   f"Leg gate ({leg_means.shape[1]}-D) routing — t-SNE{title_dir}\n"
                   f"terrain $\\eta^2$={eta_leg:.2f}")
        legend_ax = None
        rect = None

    if legend_ax is not None:
        # bidir: shared terrain legend below the two panels
        handles, leg_labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc="lower center",
                   ncol=min(8, len(leg_labels)), fontsize=6, frameon=False,
                   markerscale=1.6, handlelength=0.6, columnspacing=1.0,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=rect)
    else:
        # single panel: place the terrain legend inside the axes, in the
        # emptiest corner (t-SNE leaves blank space matplotlib can find).
        ax.legend(loc="best", ncol=2, fontsize=5.5, frameon=True,
                  framealpha=0.75, edgecolor="0.7", markerscale=1.3,
                  handlelength=0.6, handletextpad=0.4, columnspacing=0.8,
                  labelspacing=0.3, borderpad=0.4)
        fig.tight_layout()
    out = out_dir / "paper_05_leg_routing.pdf"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Plot 06 — Wheel gate per-expert violins
# ---------------------------------------------------------------------------

def plot_wheel_violin(raw, summary, valid_mask, out_dir, merge_directions=False,
                      dir_encoding="panels"):
    """Per-expert violin view of wheel-expert differentiation by terrain.

    One panel per wheel expert (W0/W1/W2); within each, a horizontal violin per
    terrain shows the distribution of that expert's gate weight across envs,
    colored by terrain with terrain names on the shared y-axis. Restricted to
    forward envs when the eval is bidirectional, since direction otherwise
    dominates and masks the terrain structure.
    """
    gate_wheel = raw["gate_wheel"].astype(np.float32)
    cmd = raw["cmd"].astype(np.float32)
    types = raw["terrain_types"]
    term_step = raw["term_step"]
    sub_names = summary["sub_terrain_names"]
    sub_per_env = env_subterrain_name(types, summary)

    E, T, K_wh = gate_wheel.shape
    step_mask = np.zeros((E, T), dtype=bool)
    for e in range(E):
        last = int(term_step[e]) if term_step[e] >= 0 else T
        step_mask[e, :max(last, 1)] = True
    cnts = step_mask.sum(axis=1, keepdims=True).clip(min=1)
    wh = (gate_wheel * step_mask[..., None]).sum(axis=1) / cnts
    vx = (cmd[..., 0] * step_mask).sum(axis=1) / cnts[:, 0]

    kept = np.isin(sub_per_env, sub_names) & valid_mask
    wh, labels, vx = wh[kept], sub_per_env[kept], vx[kept]
    if wh.shape[0] < 5:
        print("[warn] too few envs for wheel-violin plot, skipping")
        return None

    note = ""
    fwd_mask_global = None  # for split/side encoding below
    if (vx > 0).any() and (vx < 0).any():
        if dir_encoding in ("shape_split", "fill_split"):
            note = " (split: top=fwd, bottom=bwd)"
            fwd_mask_global = vx > 0
        elif dir_encoding == "shape_side":
            note = " (side-by-side: fwd | bwd per terrain)"
            fwd_mask_global = vx > 0
        elif dir_encoding == "fill_overlay":
            note = " (overlay: filled=fwd, outline=bwd)"
            fwd_mask_global = vx > 0
        elif merge_directions or dir_encoding == "merged":
            note = " (fwd + bwd merged)"
        else:
            fmask = vx > 0
            wh, labels = wh[fmask], labels[fmask]
            note = " (forward envs)"

    present = [n for n in sub_names if (labels == n).any()]
    disp = []
    for n in present:
        s = short_terrain_label(n)
        disp.append(s[:1].upper() + s[1:])
    nT = len(present)
    expert_names = [f"W{i}" for i in range(K_wh)]

    cmap = plt.cm.tab20
    n_sub = len(sub_names)

    def _tcolor(name):
        return cmap(sub_names.index(name) / max(1, n_sub - 1))

    # compact single-column figure: 3 panels kept side by side
    fig, axes = plt.subplots(1, K_wh, figsize=(3.5, 2.4), sharey=True)
    if K_wh == 1:
        axes = [axes]

    def _style_bodies(parts, color, alpha=0.78):
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("0.3")
            body.set_linewidth(0.4)
            body.set_alpha(alpha)
        for key in ("cmeans", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("0.3")
                parts[key].set_linewidth(0.6)

    for j, ax in enumerate(axes):
        if fwd_mask_global is not None and dir_encoding == "shape_side":
            # Two violins per terrain, fwd above, bwd just below — half spacing
            for ti, n in enumerate(present):
                m_t = labels == n
                fdata = wh[m_t & fwd_mask_global][:, j]
                bdata = wh[m_t & ~fwd_mask_global][:, j]
                base = ti + 1
                if len(fdata):
                    pf = ax.violinplot([fdata], positions=[base - 0.18], vert=False,
                                       showmeans=True, widths=0.30)
                    _style_bodies(pf, _tcolor(n), alpha=0.85)
                if len(bdata):
                    pb = ax.violinplot([bdata], positions=[base + 0.18], vert=False,
                                       showmeans=True, widths=0.30)
                    _style_bodies(pb, _tcolor(n), alpha=0.45)
        elif fwd_mask_global is not None and dir_encoding == "fill_overlay":
            # Overlay: per terrain row, draw fwd as filled violin + bwd as outline-only
            for ti, n in enumerate(present):
                m_t = labels == n
                fdata = wh[m_t & fwd_mask_global][:, j]
                bdata = wh[m_t & ~fwd_mask_global][:, j]
                base = ti + 1
                # fwd: filled
                if len(fdata):
                    pf = ax.violinplot([fdata], positions=[base], vert=False,
                                       showmeans=True, widths=0.85)
                    for body in pf["bodies"]:
                        body.set_facecolor(_tcolor(n))
                        body.set_edgecolor(_tcolor(n))
                        body.set_linewidth(0.6)
                        body.set_alpha(0.75)
                    for key in ("cmeans", "cbars", "cmins", "cmaxes"):
                        if key in pf:
                            pf[key].set_color("0.3")
                            pf[key].set_linewidth(0.6)
                # bwd: outline only, overlay on same row
                if len(bdata):
                    pb = ax.violinplot([bdata], positions=[base], vert=False,
                                       showmeans=True, widths=0.85)
                    for body in pb["bodies"]:
                        body.set_facecolor("none")
                        body.set_edgecolor(_tcolor(n))
                        body.set_linewidth(1.2)
                        body.set_alpha(1.0)
                    # Match fwd's helper-line styling but dashed for bwd so the two
                    # means (fwd solid / bwd dashed) are visually distinguishable.
                    if "cmeans" in pb:
                        pb["cmeans"].set_color("0.3")
                        pb["cmeans"].set_linewidth(0.6)
                        pb["cmeans"].set_linestyles("dashed")
                    for key in ("cbars", "cmins", "cmaxes"):
                        if key in pb:
                            pb[key].set_visible(False)
        elif fwd_mask_global is not None and dir_encoding in ("shape_split", "fill_split"):
            # Split violin — manually plot fwd (top half) and bwd (bottom half) of each terrain row
            for ti, n in enumerate(present):
                m_t = labels == n
                fdata = wh[m_t & fwd_mask_global][:, j]
                bdata = wh[m_t & ~fwd_mask_global][:, j]
                base = ti + 1
                # Top half violin = fwd (clip path to upper half of position)
                if len(fdata):
                    pf = ax.violinplot([fdata], positions=[base], vert=False,
                                       showmeans=False, widths=0.85)
                    for body in pf["bodies"]:
                        v = body.get_paths()[0].vertices
                        v[:, 1] = np.clip(v[:, 1], base, base + 0.5)  # top half only
                        body.set_facecolor(_tcolor(n))
                        body.set_edgecolor("0.3")
                        body.set_linewidth(0.4)
                        body.set_alpha(0.85)
                    for key in ("cmeans", "cbars", "cmins", "cmaxes"):
                        if key in pf: pf[key].set_visible(False)
                if len(bdata):
                    pb = ax.violinplot([bdata], positions=[base], vert=False,
                                       showmeans=False, widths=0.85)
                    for body in pb["bodies"]:
                        v = body.get_paths()[0].vertices
                        v[:, 1] = np.clip(v[:, 1], base - 0.5, base)  # bottom half only
                        body.set_facecolor(_tcolor(n))
                        body.set_edgecolor("0.3")
                        body.set_linewidth(0.4)
                        body.set_alpha(0.45)
                    for key in ("cmeans", "cbars", "cmins", "cmaxes"):
                        if key in pb: pb[key].set_visible(False)
        else:
            data = [wh[labels == n][:, j] for n in present]
            parts = ax.violinplot(data, positions=range(1, nT + 1), vert=False,
                                  showmeans=True, widths=0.85)
            for body, n in zip(parts["bodies"], present):
                body.set_facecolor(_tcolor(n))
                body.set_edgecolor("0.3")
                body.set_linewidth(0.4)
                body.set_alpha(0.78)
            for key in ("cmeans", "cbars", "cmins", "cmaxes"):
                if key in parts:
                    parts[key].set_color("0.3")
                    parts[key].set_linewidth(0.6)
        col = wh[:, j]  # per-panel x-range, tight to this expert's own data
        cpad = (float(col.max()) - float(col.min())) * 0.12
        ax.set_xlim(float(col.min()) - cpad, float(col.max()) + cpad)
        ax.set_title(expert_names[j], fontsize=7)
        ax.tick_params(labelsize=4.5)
        ax.locator_params(axis="x", nbins=4)
    # terrain labels on the shared y-axis (only the first panel carries them)
    axes[0].set_yticks(range(1, nT + 1))
    axes[0].set_yticklabels(disp, fontsize=5)
    axes[0].set_ylim(0.4, nT + 0.6)
    axes[0].invert_yaxis()  # first terrain at the top
    # center the x-label under the whole figure rather than just the middle panel
    fig.supxlabel("gate weight", fontsize=7, y=0.02)
    fig.suptitle("Wheel expert weight distribution per terrain", fontsize=7)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = out_dir / "paper_06_wheel_violin.pdf"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Plot 03 — Velocity tracking (per terrain, violin)
# ---------------------------------------------------------------------------

def plot_velocity_tracking(raw, summary, valid_mask, out_dir):
    cmd = raw["cmd"].astype(np.float32)               # (E,T,3) [vx,vy,wz]
    actual = raw["actual_vel"].astype(np.float32)     # (E,T,3)
    types = raw["terrain_types"]
    term_step = raw["term_step"]
    sub_names = summary["sub_terrain_names"]
    sub_per_env = env_subterrain_name(types, summary)

    E, T, _ = cmd.shape
    valid_step_mask = np.zeros((E, T), dtype=bool)
    for e in range(E):
        last = int(term_step[e]) if term_step[e] >= 0 else T
        valid_step_mask[e, :last] = True

    err = actual[..., 0] - cmd[..., 0]   # vx tracking error
    abs_err = np.abs(err)

    # Per-env MAE
    sums = (abs_err * valid_step_mask).sum(axis=1)
    counts = valid_step_mask.sum(axis=1).clip(min=1)
    per_env_mae = sums / counts

    groups = []
    labels = []
    for name in sub_names:
        m = (sub_per_env == name) & valid_mask
        if m.any():
            groups.append(per_env_mae[m])
            labels.append(short_terrain_label(name))

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    parts = ax.violinplot(groups, showmeans=False, showmedians=True,
                          widths=0.75)
    for body in parts["bodies"]:
        body.set_facecolor(ACCENT)
        body.set_edgecolor("#0b3d7a")
        body.set_alpha(0.55)
        body.set_linewidth(0.4)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("#0b3d7a")
            parts[key].set_linewidth(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("MAE of $v_x$ (m/s)")
    ax.set_title("Linear-velocity tracking error per terrain")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc",
            alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="both", bottom=False)

    out = out_dir / "paper_03_velocity_tracking.pdf"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def latest_run(base: Path) -> Path:
    runs = sorted(base.glob("*/raw.npz"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not runs:
        sys.exit(f"[err] no raw.npz under {base}")
    return runs[0].parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None,
                   help="path to a specific eval run directory containing raw.npz "
                        "and summary.json; default = newest under "
                        "logs/moe_eval/split_moe_teacher_parallel/")
    p.add_argument("--out_dir", type=str, default=None,
                   help="output dir; default = <data_dir>/plots_v2/")
    p.add_argument("--success_dist", type=float, default=None,
                   help="override success threshold (m); default = summary value")
    p.add_argument("--exclude_terrains", type=str,
                   default="hurdle_pole,hurdle_board",
                   help="comma-separated sub-terrain names to drop from all "
                        "plots. Default keeps only the canonical 'wall' hurdle "
                        "and drops the thin variants (pole, board).")
    p.add_argument("--level_cap", type=str,
                   default="stepping_stones:19,pit:19",
                   help="per-terrain max difficulty row (inclusive); envs of "
                        "that sub-terrain on higher rows are dropped from all "
                        "plots. Format 'name:row,name:row'. Empty disables. "
                        "Default caps gap/pit at L15-19 (above is mostly "
                        "failure noise).")
    p.add_argument("--no_tsne", action="store_true",
                   help="skip t-SNE plot (sklearn dependency, slow)")
    p.add_argument("--merge_directions", action="store_true",
                   help="Bidirectional eval default: paper_05 leg-routing splits "
                        "into forward/backward panels and paper_06 wheel violin "
                        "uses forward-only envs. With this flag, both plots pool "
                        "forward + backward into a single view colored by terrain.")
    p.add_argument("--wheel_dir_encoding", type=str, default=None,
                   choices=[None, "panels", "merged", "shape_split", "shape_side",
                            "fill_split", "fill_overlay"],
                   help="Override --dir_encoding just for paper_06 wheel violin. "
                        "Useful when leg-routing (paper_05) and wheel violin want "
                        "different visual styles (e.g. fill_overlay for paper_05 "
                        "but merged for paper_06). Defaults to --dir_encoding.")
    p.add_argument("--dir_encoding", type=str, default="panels",
                   choices=["panels", "merged", "shape_split", "shape_side",
                            "fill_split", "fill_overlay"],
                   help="How to encode forward/backward direction in paper_05 and "
                        "paper_06 when eval is bidirectional. panels=default (fwd/bwd "
                        "in separate panels for tsne, fwd-only for violin). "
                        "merged=pool fwd+bwd by terrain only. shape_split=tsne uses "
                        "marker shape (●/▲), violin uses split halves. "
                        "shape_side=same tsne, violin uses side-by-side pair per "
                        "terrain. fill_split=tsne uses filled/open markers, violin "
                        "uses split halves. fill_overlay=tsne fill+open, violin "
                        "overlays bwd (outline-only) on top of fwd (filled) per row. "
                        "Overrides --merge_directions when set.")
    args = p.parse_args()

    if args.data_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        base = repo_root / "logs" / "moe_eval" / "split_moe_teacher_parallel"
        data_dir = latest_run(base)
    else:
        data_dir = Path(args.data_dir)

    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "plots_v2"
    print(f"[info] data_dir = {data_dir}")
    print(f"[info] out_dir  = {out_dir}")

    raw, summary = load_run(data_dir)
    success_dist = args.success_dist if args.success_dist is not None \
                   else float(summary["success_dist"])

    success_mask, displacement, valid_mask = compute_success(raw, success_dist)

    sub_per_env = env_subterrain_name(raw["terrain_types"], summary)

    # Apply terrain exclusion: drop envs whose sub-terrain is in the exclude
    # list, and also drop those names from summary so they don't appear in
    # any plot.
    exclude = set(s.strip() for s in args.exclude_terrains.split(",")
                  if s.strip())
    if exclude:
        keep_env = ~np.isin(sub_per_env, list(exclude))
        n_dropped_envs = int((~keep_env).sum())
        valid_mask = valid_mask & keep_env
        summary["sub_terrain_names"] = [
            n for n in summary["sub_terrain_names"] if n not in exclude
        ]
        print(f"[info] excluded terrains: {sorted(exclude)} "
              f"({n_dropped_envs} envs dropped)")

    # Apply per-terrain difficulty cap: drop envs of the named sub-terrains
    # whose difficulty row exceeds the cap. The terrain still appears in plots,
    # just truncated — high-difficulty gap/pit are mostly failures that only
    # add noise to the aggregates.
    caps = {}
    for tok in args.level_cap.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, row = tok.partition(":")
        caps[name.strip()] = int(row)
    if caps:
        levels = raw["terrain_levels"]
        cap_drop = np.zeros(len(levels), dtype=bool)
        for name, max_row in caps.items():
            cap_drop |= (sub_per_env == name) & (levels > max_row)
        n_cap = int((cap_drop & valid_mask).sum())
        valid_mask = valid_mask & ~cap_drop
        print(f"[info] level cap {caps}: {n_cap} envs dropped (row > cap)")

    n_total = len(displacement)
    n_valid = int(valid_mask.sum())
    n_succ = int((success_mask & valid_mask).sum())
    print(f"[info] total envs       : {n_total}")
    print(f"[info] valid (|dx|<={MAX_DISPL}m, terrain kept): {n_valid} "
          f"({100*n_valid/n_total:.1f}%)")
    print(f"[info] successes        : {n_succ}  "
          f"({100*n_succ/max(n_valid,1):.1f}% of valid)")
    if valid_mask.any():
        print(f"[info] displ stats (valid only): "
              f"min={displacement[valid_mask].min():.2f}m  "
              f"max={displacement[valid_mask].max():.2f}m  "
              f"mean={displacement[valid_mask].mean():.2f}m")

    # Mask success to valid (compute_success may flag direction=0 invalid)
    success_mask = success_mask & valid_mask

    f1, overall = plot_success_heatmap(raw, summary, success_mask, valid_mask, out_dir)
    print(f"[done] {f1}")
    f2 = plot_routing_specialization(raw, summary, valid_mask, out_dir)
    print(f"[done] {f2}")
    f3 = plot_velocity_tracking(raw, summary, valid_mask, out_dir)
    print(f"[done] {f3}")
    if not args.no_tsne:
        f4 = plot_gate_tsne(raw, summary, valid_mask, out_dir)
        if f4:
            print(f"[done] {f4}")
        f5 = plot_leg_routing(raw, summary, valid_mask, out_dir,
                              merge_directions=args.merge_directions,
                              dir_encoding=args.dir_encoding)
        if f5:
            print(f"[done] {f5}")
    wheel_enc = args.wheel_dir_encoding or args.dir_encoding
    f6 = plot_wheel_violin(raw, summary, valid_mask, out_dir,
                           merge_directions=args.merge_directions,
                           dir_encoding=wheel_enc)
    if f6:
        print(f"[done] {f6}")


if __name__ == "__main__":
    main()
