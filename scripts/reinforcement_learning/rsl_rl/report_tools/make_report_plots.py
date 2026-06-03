"""Generate comparison plots for the SOTA course-eval report (reads course_results.json)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

REPORT = "/home/ouge/Software/rl_training/logs/moe_eval/course_sota_report"
PLOTS = f"{REPORT}/plots"
os.makedirs(PLOTS, exist_ok=True)
D = json.load(open(f"{REPORT}/data/course_results.json"))

LVLS = D["levels"]
PATCH = D["patch_names"]
# display order + labels; full first & highlighted
ORDER = ["full", "mlp_baseline", "A2", "A1", "B1", "A3", "B2"]
LABEL = {"full": "full (SplitMoE)", "mlp_baseline": "MLP baseline",
         "A1": "A1 · single gate", "A2": "A2 · shared critic",
         "A3": "A3 · no L_sym", "B1": "B1 · no L_bal", "B2": "B2 · blind vision"}
LVL_LABEL = {"easy": "Easy\n(d=0.40)", "med": "Med\n(d=0.50)",
             "hard": "Hard\n(d=0.70)", "extreme": "Extreme\n(d=0.95)"}
COL = {"full": "#d62728", "mlp_baseline": "#1f77b4", "A2": "#ff7f0e",
       "A1": "#2ca02c", "B1": "#9467bd", "A3": "#8c564b", "B2": "#7f7f7f"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "svg.fonttype": "none"})

def get(v, l, key):
    c = D["cells"].get(f"{v}|{l}")
    return c.get(key, np.nan) if c else np.nan

def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(f"{PLOTS}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

# ---- 1 & 2: grouped bars binary + mean, per level ----
def grouped_bars(metric, std_key, title, fname, ylabel):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    nL = len(LVLS); nV = len(ORDER); w = 0.8 / nV
    x = np.arange(nL)
    for i, v in enumerate(ORDER):
        vals = [get(v, l, metric) for l in LVLS]
        errs = [get(v, l, std_key) for l in LVLS]
        bars = ax.bar(x + (i - nV/2 + 0.5) * w, vals, w, yerr=errs, capsize=2,
                      label=LABEL[v], color=COL[v],
                      edgecolor="black" if v == "full" else "none",
                      linewidth=1.4 if v == "full" else 0,
                      zorder=3 if v == "full" else 2, alpha=1.0 if v == "full" else 0.85)
        if v == "full":
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, val + 0.02, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold", color=COL["full"])
    ax.set_xticks(x); ax.set_xticklabels([LVL_LABEL[l] for l in LVLS])
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.08); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", ls=":", lw=0.5, color="#ccc", zorder=0)
    save(fig, fname)

grouped_bars("binary", "binary_std", "Course completion rate by difficulty (binary_complete)",
             "01_binary_by_level", "completion rate")
grouped_bars("mean", "mean_std", "Mean progress by difficulty (distance fraction)",
             "02_mean_by_level", "mean progress_ratio")

# ---- 3: overall robustness (avg binary across 4 levels) ----
fig, ax = plt.subplots(figsize=(7, 4))
ov = [(v, np.nanmean([get(v, l, "binary") for l in LVLS])) for v in ORDER]
ov.sort(key=lambda t: -t[1])
names = [LABEL[v] for v, _ in ov]; vals = [s for _, s in ov]
cols = [COL[v] for v, _ in ov]
bars = ax.barh(range(len(ov)), vals, color=cols,
               edgecolor=["black" if v == "full" else "none" for v, _ in ov],
               linewidth=[1.4 if v == "full" else 0 for v, _ in ov])
ax.set_yticks(range(len(ov))); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("mean completion across all 4 difficulties"); ax.set_xlim(0, 1)
for i, (v, s) in enumerate(ov):
    ax.text(s + 0.01, i, f"{s:.3f}", va="center", fontsize=9,
            fontweight="bold" if v == "full" else "normal")
ax.set_title("Overall robustness (↑ better)", fontsize=13, fontweight="bold")
ax.grid(axis="x", ls=":", lw=0.5, color="#ccc")
save(fig, "03_overall_robustness")

# ---- 4: per-patch completion heatmaps, one per level ----
for l in LVLS:
    M = np.array([get(v, l, "per_patch") if D["cells"].get(f"{v}|{l}") else [np.nan]*12
                  for v in ORDER], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(PATCH, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LABEL[v] for v in ORDER], fontsize=9)
    for yi in range(len(ORDER)):
        for xi in range(12):
            val = M[yi, xi]
            if not np.isnan(val):
                ax.text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if 0.25 < val < 0.85 else "white")
    ax.set_title(f"Per-obstacle pass rate — {l} (d={ {'easy':0.40,'med':0.50,'hard':0.70,'extreme':0.95}[l] })",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="pass rate")
    save(fig, f"04_perpatch_{l}")

# ---- 5: the A2 leap-cheat illustration (extreme: mean vs binary) ----
fig, ax = plt.subplots(figsize=(7, 4.2))
vs = ["full", "A2", "mlp_baseline", "A1"]
xm = [get(v, "extreme", "mean") for v in vs]
xb = [get(v, "extreme", "binary") for v in vs]
x = np.arange(len(vs)); w = 0.38
ax.bar(x - w/2, xm, w, label="mean progress (distance)", color="#bbbbbb", edgecolor="black", lw=0.5)
ax.bar(x + w/2, xb, w, label="binary completion (finished)", color="#d62728", edgecolor="black", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([LABEL[v] for v in vs], fontsize=9)
ax.set_ylabel("rate"); ax.set_ylim(0, 1.05)
ax.set_title("Extreme: 'distance' vs 'finished' — A2's leap inflates distance, not completion",
             fontsize=11.5, fontweight="bold")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", ls=":", lw=0.5, color="#ccc")
for i, v in enumerate(vs):
    ax.text(i - w/2, xm[i] + 0.01, f"{xm[i]:.2f}", ha="center", fontsize=7.5)
    ax.text(i + w/2, xb[i] + 0.01, f"{xb[i]:.2f}", ha="center", fontsize=7.5, color="#d62728")
save(fig, "05_a2_leap_cheat")

print("plots written to", PLOTS)
print(os.listdir(PLOTS))
