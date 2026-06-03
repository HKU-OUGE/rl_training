"""Publication bar chart: course completion rate, variants grouped by difficulty.
Saves a vector PDF into both papers' figures/ dirs."""
import json, numpy as np, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("/home/ouge/Software/rl_training/logs/moe_eval/course_sota_report/data/course_results.json"))
LVLS = ["easy", "med", "hard", "extreme"]
LVL_LBL = ["Easy\n(d=0.40)", "Med\n(d=0.50)", "Hard\n(d=0.70)", "Extreme\n(d=0.95)"]
ORDER = ["full", "mlp_baseline", "A1", "A2", "B1", "A3", "B2"]
LBL = {"full": "SplitMoE (full)", "mlp_baseline": "MLP baseline",
       "A1": "A1 single gate", "A2": "A2 shared critic",
       "B1": "B1 no $\\mathcal{L}_{bal}$", "A3": "A3 no $\\mathcal{L}_{sym}$", "B2": "B2 blind"}
COL = {"full": "#d62728", "mlp_baseline": "#1f77b4", "A1": "#2ca02c", "A2": "#ff7f0e",
       "B1": "#9467bd", "A3": "#8c564b", "B2": "#7f7f7f"}

def g(v, l, k):
    c = D["cells"].get(f"{v}|{l}"); return (c.get(k, np.nan) if c else np.nan)

plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.0, 2.9))
nL, nV = len(LVLS), len(ORDER); w = 0.8 / nV; x = np.arange(nL)
for i, v in enumerate(ORDER):
    vals = [g(v, l, "binary") * 100 for l in LVLS]
    errs = [g(v, l, "binary_std") * 100 for l in LVLS]
    isf = v == "full"
    bars = ax.bar(x + (i - nV/2 + 0.5) * w, vals, w, yerr=errs, capsize=1.5,
                  label=LBL[v], color=COL[v], edgecolor="black" if isf else "none",
                  linewidth=1.2 if isf else 0, zorder=3 if isf else 2,
                  alpha=1.0 if isf else 0.88, error_kw={"lw": 0.6})
    if isf:
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, val + 1.5, f"{val:.0f}", ha="center",
                    va="bottom", fontsize=6.5, fontweight="bold", color=COL["full"])
ax.set_xticks(x); ax.set_xticklabels(LVL_LBL, fontsize=8)
ax.set_ylabel("course completion (\\%)"); ax.set_ylim(0, 108)
ax.legend(ncol=4, fontsize=6.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30),
          columnspacing=1.0, handlelength=1.2)
ax.grid(axis="y", ls=":", lw=0.4, color="#ccc", zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()

out = "/tmp/course_ablation_bars.pdf"
plt.savefig(out, bbox_inches="tight"); plt.close(fig)
for d in ["/home/ouge/Desktop/SplitMoE/figures", "/home/ouge/Desktop/SplitMoE_EN/figures"]:
    shutil.copy(out, f"{d}/course_ablation_bars.pdf")
    print("copied ->", d)
print("done")
