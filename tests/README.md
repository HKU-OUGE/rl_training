# tests/

Unit and integration tests for `moe_terrain.py` (the SplitMoE actor-critic + PPO).

## Quick start

```bash
bash tests/run_all.sh                                                 # all
$PYTHON tests/test_mirror_fb.py                                       # one file
$PYTHON tests/test_actor_critic_forward.py                            # one file
```

`$PYTHON` defaults to `/home/ouge/miniconda3/envs/env_isaaclab/bin/python`.

## File layout

| File | Coverage |
|---|---|
| `conftest.py` | Shared infra: path constants, IsaacLab stub injection, fake-obs builder, `make_actor_critic()` / `make_ppo()` factories |
| `test_mirror_fb.py` | Reference implementation of `_mirror_obs` (L-R) and `_mirror_obs_fb` (F-B). Verifies prefix flips, joint swap+neg, wheel signs, elevation flip axis, scan channel swap, involution, Klein-4 commutativity, C2 = LR ∘ FB. 8 tests, ~50 assertions. |
| `test_mirror_fb_integration.py` | Extracts `_mirror_obs` / `_mirror_obs_fb` method bodies via AST from the actual source file, executes them with a mocked `self`, and compares outputs to the reference impl across multiple batch sizes / history lengths / seeds. 36 assertions. |
| `test_moe_terrain_components.py` | Tier 1: `EmpiricalNormalization` (Welford), `MLP`, `ElevationAE`, `MultiLayerScanAE`, `DepthAE`, `ProprioVAE` shape/mode tests. Tier 2: joint-name-list ↔ swap-index consistency; mask signs ↔ URDF `init_state` consistency. Tier 3: static checks of bypass paths and PPO step order. 14 tests. |
| `test_actor_critic_forward.py` | Instantiates a real `SplitMoEActorCritic` via stubbed IsaacLab imports. Tests `act()`, `act_inference()`, `evaluate()` shapes; hidden-state lifecycle (init, evolve, reset(dones), shape stability); PPO piggyback (`SplitMoEPPO.act` updates layer 1); gradient flow; aux loss path; sym MSE non-zero at random init. 12 tests. |
| `test_sym_loss_training.py` | Trains a small actor with sym MSE alone for 150 steps. Verifies sym MSE drops >80%; zero-output policy gives MSE=0 (trivial fixed point); zero-obs sanity (mirror is identity). 3 tests. |

## What gets imported

`conftest.install_isaaclab_stubs()` injects fake modules into `sys.modules` so
`moe_terrain.py` can be imported without launching the Omniverse Kit
(`pxr` / IsaacLab assets). The stubs cover:

- `isaaclab.utils.configclass` → no-op decorator
- `isaaclab_rl.rsl_rl.RslRl*Cfg` → permissive base class accepting any kwargs

`rsl_rl` itself is real (it's pure PyTorch). `SplitMoEActorCritic` and `SplitMoEPPO`
work fully with stubbed IsaacLab — only the cfg classes (which we don't
exercise directly) depend on the stubbed IsaacLab pieces.

## Adding a new test

```python
# tests/test_my_new_feature.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import make_actor_critic, make_ppo, make_fake_obs

def test_something():
    model, obs = make_actor_critic(B=4, latent_dim=32)
    ...
```

Then add the file name to `TESTS=(...)` in `run_all.sh`.

## What is NOT covered

These would require launching `SimulationApp`:
- Distillation path (`SymmetricMoEDistillation`, `SplitMoEStudentTeacher`) — explicitly out of scope
- Full env stepping (terrain generation, contact forces, randomization)
- Reward function correctness on rolled-out trajectories
- Multi-GPU training synchronization
