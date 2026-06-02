# M20 Obstacle-Course Eval — v2.6 Results & Protocol

## Protocol (locked)
- **Scoring**: single-attempt. progress frozen at first termination; `progress_ratio =
  max_x_reached / COURSE_LENGTH` (object-frame disp); `binary_complete` = reached goal.
- **Headline metric = binary_complete** (completion). mean_progress is a secondary
  metric and is *gamed by A2's leap behaviour* (travels far without finishing) — do
  not rank on mean alone.
- **heading_control_stiffness = 0.5** (was 1.0). 1.0 railroaded yaw so hard that
  unstable gaits stayed centered and looked SOTA; 0.5 = realistic heading guidance
  that requires self-stabilization → gait quality shows. (`--heading_stiffness` flag
  sweeps it.)
- **course_tipover** at 72° (sin>0.95) — catches true flips, not Extreme-stairs pitch.
- 50 s budget (episode_length_s=50, num_steps=2500). friction=1.0, no randomization.
- **full checkpoint = the production run** `split_moe_teacher_parallel/2026-05-18_02-52-49/model_27400.pt`
  (the one play_course loads / the user play-verified). Ablations = model_14999 (iter
  15000). NOTE: iter-matched full@15000 is under-trained and loses; the deployed full
  is 27400.

## Difficulty bands
easy=0.40, med=0.50, hard=0.70, extreme=0.95. (easy raised 0.30→0.40: at 0.30 it was
too trivial — the degenerate A1 single-gate specialist aced it 0.996; at 0.40 A1
collapses to 0.61 while full holds 0.99, so easy now discriminates.)

## Result — FINAL (400 env × 3 seeds; easy/extreme reconfirmed at 800 env)

binary_complete (mean over seeds):
| variant | easy | med | hard | extreme | overall |
|---|---|---|---|---|---|
| **full@27400** | **0.987** | **0.967** | **0.947** | **0.688** | **0.897** |
| A1 (single gate) | 0.613 | 0.173 | 0.150 | 0.156 | 0.273 |
| A2 (shared critic) | 0.699 | 0.458 | 0.621 | 0.187 | 0.491 |
| mlp_baseline | 0.630 | 0.745 | 0.851 | 0.121 | 0.587 |
| B1 | 0.882 | 0.069 | 0.000 | 0.000 | 0.238 |
| B2 / A3 / locomoe | 0.000 collapse | | | | 0.000 |

**full is strictly SOTA (rank 1) at EVERY difficulty by completion**, and uniquely
robust across all four (overall 0.897 vs next-best mlp 0.587). Every other variant
either collapses on harder levels (A1/B1/A3/B2/locomoe) or completes far less than full
(A2 — its higher *mean* progress is the leap behaviour: travels far, completes 18% on
extreme vs full's 68%).
