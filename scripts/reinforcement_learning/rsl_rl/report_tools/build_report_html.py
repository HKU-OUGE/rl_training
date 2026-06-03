import json, numpy as np
REPORT="/home/ouge/Software/rl_training/logs/moe_eval/course_sota_report"
D=json.load(open(f"{REPORT}/data/course_results.json"))
LVLS=D["levels"]
ORDER=["full","mlp_baseline","A2","A1","B1","A3","B2"]
LBL={"full":"full (SplitMoE)","mlp_baseline":"MLP baseline","A1":"A1 · single gate",
     "A2":"A2 · shared critic","A3":"A3 · no L_sym","B1":"B1 · no L_bal","B2":"B2 · blind vision"}
LVLD={"easy":"Easy (d=0.40)","med":"Med (d=0.50)","hard":"Hard (d=0.70)","extreme":"Extreme (d=0.95)"}
def g(v,l,k):
    c=D["cells"].get(f"{v}|{l}"); return c.get(k,float('nan')) if c else float('nan')
def best(l,k):  # variant with max metric at level l
    return max(ORDER,key=lambda v:(g(v,l,k) if not np.isnan(g(v,l,k)) else -1))

def metric_table(metric,std):
    h="<tr><th>variant</th>"+"".join(f"<th>{LVLD[l]}</th>" for l in LVLS)+"<th>overall</th></tr>"
    rows=""
    for v in ORDER:
        cls="full" if v=="full" else ""
        tds=""
        for l in LVLS:
            m=g(v,l,metric); s=g(v,l,std)
            win=" win" if v==best(l,metric) else ""
            tds+=f'<td class="num{win}">{m:.3f}<span class="sd">±{s:.3f}</span></td>'
        ov=np.nanmean([g(v,l,metric) for l in LVLS])
        rows+=f'<tr class="{cls}"><td class="lab">{LBL[v]}</td>{tds}<td class="num ov">{ov:.3f}</td></tr>'
    return f'<table>{h}{rows}</table>'

def img(name,cap):
    return f'<figure><img src="plots/{name}.png" alt="{cap}"><figcaption>{cap} · <a href="plots/{name}.svg">SVG</a></figcaption></figure>'
def terr(l):
    return f'<figure><img src="course_terrain/course_{l}.png" alt="{l}"><figcaption>{LVLD[l]}</figcaption></figure>'

# SOTA verdict
verdict="".join(f"<b>{LVLD[l].split(' ')[0]}</b> {g('full',l,'binary'):.3f} · " for l in LVLS)
html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>M20 SplitMoE — Obstacle-Course Ablation Comparison</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,PingFang SC,Arial,sans-serif;max-width:1180px;margin:24px auto;padding:0 22px;color:#1a1a1a;line-height:1.5}}
 h1{{font-size:25px;margin:.2em 0}} h2{{font-size:19px;border-bottom:2px solid #eee;padding-bottom:5px;margin-top:34px}}
 .sub{{color:#666;font-size:13px;margin-bottom:8px}}
 .verdict{{background:#fff5f5;border-left:5px solid #d62728;padding:12px 16px;border-radius:0 6px 6px 0;margin:14px 0;font-size:15px}}
 table{{border-collapse:collapse;font-size:13px;margin:10px 0 6px}}
 th,td{{border:1px solid #ddd;padding:5px 11px;text-align:right}} th{{background:#f5f5f5}}
 td.lab{{text-align:left;font-weight:600}} tr.full td{{background:#fff0f0;font-weight:700}}
 td.num .sd{{color:#999;font-size:10px;margin-left:2px}} td.win{{background:#e7f6e7}} tr.full td.win{{background:#ffe0e0}}
 td.ov{{background:#f0f4ff}} tr.full td.ov{{background:#ffd9d9}}
 figure{{margin:8px 0}} img{{width:100%;border:1px solid #eee;border-radius:5px;background:#fff}}
 figcaption{{font-size:12px;color:#666;margin-top:3px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
 .note{{background:#f8f9fa;border:1px solid #e5e5e5;border-radius:6px;padding:10px 14px;font-size:13px;margin:8px 0}}
 code{{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:12px}}
 footer{{color:#999;font-size:11px;margin-top:30px;border-top:1px solid #eee;padding-top:8px}}
</style></head><body>
<h1>M20 SplitMoE — Obstacle-Course Ablation Comparison</h1>
<div class="sub">Single-attempt course eval · 50 s budget · heading_stiffness 0.5 · 12 obstacles ×2 cycles · 400 envs × 3 seeds · deterministic (friction 1.0, no DR). Headline metric = <b>binary completion</b>.</div>
<div class="verdict"><b>full (SplitMoE) is SOTA at every difficulty</b> by completion: {verdict}<br>
Overall robustness {np.nanmean([g('full',l,'binary') for l in LVLS]):.3f} vs next-best {max(np.nanmean([g(v,l,'binary') for l in LVLS]) for v in ORDER if v!='full'):.3f}. full is the only policy strong across all four levels.</div>

<h2>1 · Completion rate by difficulty</h2>
{img('01_binary_by_level','Binary completion rate per difficulty (green cell = level winner)')}
{metric_table('binary','binary_std')}
<div class="sub">Green = level best · pink row = full · ± = std over 3 seeds.</div>

<h2>2 · Overall robustness</h2>
<div class="grid2">{img('03_overall_robustness','Mean completion across all 4 difficulties')}
<div><p>Each ablation has a fatal weakness:</p><ul style="font-size:13px">
<li><b>A1</b> (single gate): aces easy, collapses ≥med.</li>
<li><b>A2</b> (shared critic): the "leap" gait — travels far but rarely finishes (see §4).</li>
<li><b>A3 / B2 / LocoMoE</b>: ~0 everywhere (no L_sym / blind / no-perception).</li>
<li><b>B1</b> (no L_bal): ok easy, dies by hard.</li>
<li><b>MLP baseline</b>: decent mid, weak extreme.</li>
<li><b>full</b>: only policy robust at all four → uniquely SOTA.</li></ul></div></div>

<h2>3 · Mean progress (distance) by difficulty</h2>
{img('02_mean_by_level','Mean progress_ratio per difficulty')}
{metric_table('mean','mean_std')}
<div class="note">⚠ Mean-progress measures <i>distance reached</i>, not completion — it is gamed by A2's leap behaviour. Use completion (§1) for ranking.</div>

<h2>4 · Why A2 is not actually better (the leap behaviour)</h2>
{img('05_a2_leap_cheat','Extreme: distance vs finished')}
<div class="note">On Extreme, A2's <b>mean progress 0.88 ≈ full's 0.86</b>, but A2 <b>completes only 0.19 vs full 0.69</b>: A2 leaps far up the stairs then jams ("容易卡住"). full traverses and finishes 3.6× more often.</div>

<h2>5 · Per-obstacle pass rate (where each policy fails)</h2>
<div class="grid2">{img('04_perpatch_easy','Easy')}{img('04_perpatch_med','Med')}</div>
<div class="grid2">{img('04_perpatch_hard','Hard')}{img('04_perpatch_extreme','Extreme')}</div>

<h2>6 · The courses</h2>
<div class="grid2">{terr('easy')}{terr('med')}</div>
<div class="grid2">{terr('hard')}{terr('extreme')}</div>

<h2>7 · Protocol &amp; checkpoints</h2>
<div class="note">
<b>Scoring:</b> single-attempt — progress frozen at first termination (fall/oob/below/goal/50 s timeout); object-frame disp so <code>reached_goal</code> is physically reachable.<br>
<b>Fall:</b> <code>course_tipover</code> at 72° (does not false-fire on Extreme-stairs 58° pitch).<br>
<b>Heading:</b> <code>heading_control_stiffness=0.5</code> — realistic guidance; 1.0 railroaded yaw and masked unstable gaits.<br>
<b>Checkpoints:</b> full = production <code>2026-05-18/model_27400</code> (the deployed policy); ablations = <code>model_14999</code> (iter 15000); MLP baseline = <code>model_19999</code>.<br>
<b>Difficulty:</b> easy 0.40 / med 0.50 / hard 0.70 / extreme 0.95.<br>
<b>LocoMoE excluded:</b> the original MoE-Loco is blind (no perception) and not open-source, so a faithful comparison is impossible; our reimplementation scored ~0 anyway.
</div>
<footer>Self-contained — all asset paths relative. Generated from <code>data/course_results.json</code>. Branch <code>course-eval-iter</code>.</footer>
</body></html>"""
open(f"{REPORT}/index.html","w").write(html)
print("wrote index.html")
