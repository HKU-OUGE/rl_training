import numpy as np, glob, json
from collections import defaultdict
VARS="full A1 A2 A3 B1 B2 mlp_baseline locomoe".split()
LVLS="easy med hard extreme".split()
SEEDS=[42,7,137]
cells=defaultdict(dict)  # (v,l) -> seed -> (mean,binary)
for f in glob.glob("logs/moe_eval/v26/*/summary.json"):
    d=json.load(open(f)); v,l,s=d["ablation"],d["level"],d["seed"]
    r=dict(np.load(f.replace("summary.json","raw.npz")))
    C=float(d["course_length"]); p=np.clip(r["max_x_reached"]/C,0,1)
    cells[(v,l)][s]=(float(p.mean()),float((p>=1).mean()))
def agg(v,l,idx):
    vals=[cells[(v,l)][s][idx] for s in SEEDS if s in cells.get((v,l),{})]
    return (np.mean(vals),np.std(vals),len(vals)) if vals else (float('nan'),0,0)
for idx,nm in [(0,"MEAN PROGRESS"),(1,"BINARY")]:
    print(f"\n=== {nm} (mean±std / {len(SEEDS)} seeds) ===")
    print(f"{'V':<13}|"+"".join(f"{l:>15}" for l in LVLS))
    print("-"*76)
    for v in VARS:
        row=[f"{v:<13}|"]
        for l in LVLS:
            m,sd,n=agg(v,l,idx)
            row.append(f"{m:>7.3f}±{sd:<5.3f}" if n else f"{'--':>14} ")
        print("".join(row))
print("\n=== IS FULL SOTA? (per level, on mean_progress avg over seeds) ===")
ok=True
for l in LVLS:
    scores=sorted(((agg(v,l,0)[0],v) for v in VARS if not np.isnan(agg(v,l,0)[0])),reverse=True)
    rank=[v for _,v in scores].index("full")+1
    fm=agg("full",l,0)[0]; win=scores[0]
    gap=win[0]-fm
    if rank==1: print(f"  {l:<8} full={fm:.3f}  rank=1  OK SOTA")
    else: print(f"  {l:<8} full={fm:.3f}  rank={rank}  ** winner={win[1]} {win[0]:.3f} (+{gap:.3f}) **"); ok=False
print(f"\n{'>>> FULL IS SOTA AT ALL LEVELS <<<' if ok else '>>> not yet <<<'}")
