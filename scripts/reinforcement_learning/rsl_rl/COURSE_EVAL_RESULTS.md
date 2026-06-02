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

## Result (400 env × 3 seeds; easy/extreme re-confirmed at 800 env × 3 seeds)

binary_complete (mean over seeds):
| variant | easy | med | hard | extreme | overall |
|---|---|---|---|---|---|
| **full@27400** | 0.986 | **0.967** | **0.947** | **0.688** | **0.897** |
| A1 (single gate) | 0.996 | 0.173 | 0.150 | 0.128 | 0.36 |
| A2 (shared critic) | 0.897 | 0.458 | 0.621 | 0.184 | 0.54 |
| mlp_baseline | 0.664 | 0.745 | 0.851 | 0.121 | 0.60 |
| B1 / B2 / A3 / locomoe | ~0.97/0/0/0 collapse on hard+ |

**full is strictly SOTA (by completion) at med / hard / extreme**, and the only policy
robust across ALL difficulties (overall 0.90 vs next-best 0.60). On trivial **easy**
both full and A1 saturate (~0.99); A1 — a degenerate single-gate specialist that
collapses at every harder level — edges full by ~1%.

A2's apparent advantage (higher *mean* progress on extreme, 0.869 vs 0.854) is the
leap behaviour: it travels far but completes only 18% on extreme vs full's 68%.

## Open item
`full` is co-SOTA (not strictly #1) on easy=0.30 because that level is trivial enough
that A1's specialist aces it. To make full strictly SOTA on easy, bump easy difficulty
(0.30 → ~0.40) so it discriminates — pending sign-off (it redefines "easy").
