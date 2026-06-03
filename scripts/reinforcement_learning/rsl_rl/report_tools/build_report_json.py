import numpy as np, glob, json, os
from collections import Counter, defaultdict

VARS = "full A1 A2 A3 B1 B2 mlp_baseline".split()  # LocoMoE dropped
LVLS = "easy med hard extreme".split()
SEEDS = [42, 7, 137]
PATCH = ["hurdle","slope","stairs","rail","stones","step_up",
         "hurdle2","slope2","stairs2","rail2","stones2","step_up2"]
ENUM = {0:"time_out",3:"bad_ori",5:"oob",6:"below_ground",10:"reached_goal",11:"tipover"}

# level -> source glob (easy uses the 0.40 re-run; others use v26)
def cell_dir(v, l, s):
    if l == "easy":
        return f"logs/moe_eval/v26easy40/{v}_s{s}"   # easy@0.40 re-run (no level in name)
    return f"logs/moe_eval/v26/{v}_{l}_s{s}"

out = {"variants": VARS, "levels": LVLS, "seeds": SEEDS, "patch_names": PATCH,
       "course_length": None, "cells": {}}

for v in VARS:
    for l in LVLS:
        per_seed_mean, per_seed_bin = [], []
        pp_acc = np.zeros(12); pp_n = 0
        term = Counter(); ff = Counter()
        for s in SEEDS:
            d = cell_dir(v, l, s)
            sf, rf = f"{d}/summary.json", f"{d}/raw.npz"
            if not (os.path.exists(sf) and os.path.exists(rf)):
                continue
            meta = json.load(open(sf)); r = dict(np.load(rf))
            C = float(meta["course_length"]); out["course_length"] = C
            p = np.clip(r["max_x_reached"] / C, 0, 1)
            per_seed_mean.append(float(p.mean()))
            per_seed_bin.append(float((p >= 1).mean()))
            pp_acc += r["per_patch_pass"].mean(axis=0); pp_n += 1
            for k in r["term_cause"].tolist(): term[ENUM.get(k, str(k))] += 1
            for k in r["first_fail_patch"].tolist():
                if k >= 0: ff[PATCH[k]] += 1
        if not per_seed_mean:
            continue
        out["cells"][f"{v}|{l}"] = {
            "mean": float(np.mean(per_seed_mean)), "mean_std": float(np.std(per_seed_mean)),
            "binary": float(np.mean(per_seed_bin)), "binary_std": float(np.std(per_seed_bin)),
            "per_patch": (pp_acc / max(pp_n, 1)).round(4).tolist(),
            "term_cause": dict(term), "first_fail": dict(ff),
            "n_seeds": len(per_seed_mean),
        }

json.dump(out, open("/tmp/course_results.json", "w"), indent=2)
print(f"wrote {len(out['cells'])} cells, COURSE_LENGTH={out['course_length']}")
# quick sanity
for l in LVLS:
    row = " ".join(f"{v}={out['cells'].get(f'{v}|{l}',{}).get('binary',float('nan')):.2f}" for v in VARS)
    print(f"  {l:<8} binary: {row}")
