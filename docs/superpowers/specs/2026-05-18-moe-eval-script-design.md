# MoE Teacher 评估脚本设计 (Rough-MoE-Teacher-Deeprobotics-M20-v0)

**日期**：2026-05-18
**任务**：为 `SplitMoEActorCritic` 策略写一对评估脚本，量化各 sub-terrain 上的成功率、专家激活分布、速度跟踪精度、奖励分解，以及 5 张 MoE 专属可视化图。
**目标策略**：`Rough-MoE-Teacher-Deeprobotics-M20-v0`，默认 checkpoint 为 `logs/moe_training/split_moe_teacher_parallel/<latest_run>/model_<latest_iter>.pt`（脚本自动解析）。

## 1. 架构总览

两个解耦的脚本：

| 脚本 | 职责 | 输出 |
|------|------|------|
| `scripts/reinforcement_learning/rsl_rl/eval_moe.py` | 启动 IsaacLab、加载 ckpt、采集 raw stats | `logs/moe_eval/<run_name>/<timestamp>/{raw.npz, summary.json}` |
| `scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py` | 纯 CPU 绘图（不依赖 sim） | `.../plots/*.png` |

解耦原因：sim 跑一次 ~5-15 分钟，调图样需要快速迭代；后处理脚本只读 `raw.npz`。

## 2. Sim 实现（eval_moe.py）

### 2.1 终端调用

```bash
python scripts/reinforcement_learning/rsl_rl/eval_moe.py \
    --task Rough-MoE-Teacher-Deeprobotics-M20-v0 \
    --num_envs 2000 \
    --num_steps 500 \
    --success_dist 4.0 \
    --cmd_vx 1.0 \
    --headless \
    [--load_run <run_dir>] [--checkpoint <model_NNN.pt>] \
    [--strict_per_terrain] [--zero_obs_noise] [--output_dir <override>]
```

`--num_steps` 与 `episode_length_s = 10.0` 联动；若改 `--num_steps`，episode_length_s 自动设为 `num_steps * dt`。

`--load_run` / `--checkpoint` 不传时，自动选 `logs/moe_training/split_moe_teacher_parallel/` 下最新 timestamped run 中 iter 最高的 `model_*.pt`。

### 2.2 地形 dispatch（默认快速路径）

**默认（推荐）**：单 sim 多 sub-terrain 并行采样。
- 地形 = `MOE_ROUGH_TERRAINS_CFG`（30 行 level × 18 列；13 种 sub-terrain，部分有重复列）
- `env_cfg.scene.terrain.terrain_generator.curriculum = False` → envs 锁在初始 cell
- `env_cfg.scene.terrain.max_init_terrain_level = num_rows - 1` → spawn 时按 level 均匀分散
- 关闭 `curriculum.terrain_levels`（设 `None`） → 评估期不 promote/demote
- 关闭命令 curriculum（`env_cfg.curriculum.command_levels_lin_vel = None`、`command_levels_ang_vel = None`）
- 2000 envs / (30 levels × 18 cols) ≈ 3.7 envs/cell；每 env 跑 1 个 episode → 每 cell 样本 3-4 个，聚合到 13 sub-terrain 后每 (level, sub-terrain) cell ~5-10 个
- 收集后按 `terrain_types` 聚合到 13 种 sub-terrain（同种 sub-terrain 的多列按 sub-terrain name 合并）

**Strict 路径**（`--strict_per_terrain`）：循环 13 种 sub-terrain，每次完全重启 IsaacLab simulation_app，分别用 `STAIR_ONLY_TEACHER_TERRAINS_CFG` 等单一 sub-terrain 的 generator。预计耗时 13× 默认路径。仅在用户需要严格隔离时启用。

### 2.2.1 Eval 协议（env_cfg overrides）

**目的**：为了让评估数据 crisp & comparable，关闭训练期所有 domain randomization 与命令随机性，统一让 robot 朝 +x 方向以 1 m/s 走，最多走 10s 或越过 4m 即终止。

#### 命令 (`env_cfg.commands.base_velocity`)
| 字段 | 训练值 | 评估值 |
|------|--------|--------|
| `heading_command` | True | **True** (保持) |
| `rel_heading_envs` | 1.0 | **1.0** (全部用 heading 模式) |
| `rel_standing_envs` | 0.02 | **0.0** (无静止 env) |
| `resampling_time_range` | (10.0, 10.0) | **(100.0, 100.0)** (绝不 mid-ep 重采样) |
| `ranges.heading` | (-π, π) | **(0.0, 0.0)** (统一朝 +x) |
| `ranges.lin_vel_x` | (-1.0, 1.0) | **(1.0, 1.0)** (固定 1.0 m/s，CLI `--cmd_vx 1.0` 可调) |
| `ranges.lin_vel_y` | (0.0, 0.0) | **(0.0, 0.0)** (已为 0) |
| `ranges.ang_vel_z` | (-1.5, 1.5) | **(0.0, 0.0)** (heading 模式下 ang_vel 由 heading 控制器计算，但 ranges 设 0 保险) |

#### Episode 长度
| 字段 | 值 |
|------|-----|
| `env_cfg.episode_length_s` | **10.0** |
| `--num_steps` 默认 | **500** (= 10s / dt=0.02s) |

#### Events 全关 (set to `None`)
- `events.randomize_rigid_body_material` (摩擦)
- `events.randomize_rigid_body_mass` (link 质量)
- `events.randomize_rigid_body_mass_base` (base 质量加偏)
- `events.randomize_rigid_body_inertia` (惯量)
- `events.randomize_com_positions` (重心)
- `events.randomize_apply_external_force_torque` (reset 时外力)
- `events.randomize_actuator_gains` (PD gains)
- `events.randomize_push_robot` (interval push)

#### Reset 几何归零 (`events.randomize_reset_base`)
```python
pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
              "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "z": (0.0, 0.0)}
velocity_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                  "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}
```
（保留 `randomize_reset_joints`，它本来就是 scale=1.0 + vel=0，确定性 reset。）

#### Termination：新增 `reached_goal` term
注入到 `env_cfg.terminations`：

```python
def _eval_reached_goal_x(env, threshold: float) -> torch.Tensor:
    # env.scene.env_origins 是 per-env spawn 锚点 (terrain dispatch 给的)
    disp_x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return disp_x >= threshold

env_cfg.terminations.reached_goal = DoneTerm(
    func=_eval_reached_goal_x,
    params={"threshold": args.success_dist},
)
```

终止时 `info["log"]["Episode_Termination/reached_goal"]` 自动出现，`term_cause` enum 加值 `4 = reached_goal`。

#### 观测噪声
**保留** 训练期的 `AdditiveUniformNoiseCfg`（policy 输入侧 noise）—— 不属于 domain randomization 范畴，是策略的正常输入协议。如需消融，可加 `--zero_obs_noise` flag（默认 off）。

### 2.3 Checkpoint 自动解析

复用 `play_moe.py:resolve_checkpoint_path` 的逻辑：
- `root = logs/moe_training/<experiment_name>`（`experiment_name = train_cfg_dict["experiment_name"]`，对 MoE Teacher 即 `split_moe_teacher_parallel`）
- 按 mtime 选最新子目录
- 子目录内按 mtime 选 `model_*.pt`

### 2.4 数据采集（per-step buffers）

所有 buffer 在 GPU 上预分配，最后一次性 `cpu().numpy()` 到 `raw.npz`：

| 数组 | 形状 | dtype | 用途 |
|------|------|-------|------|
| `terrain_types` | `[N]` | int32 | sub-terrain index (0..17)，常量 |
| `terrain_levels` | `[N]` | int32 | level (0..29)，常量 |
| `sub_terrain_names` | `[13]` | str list | summary.json |
| `alive_mask` | `[N, T]` | bool | env 在第 t 步是否仍在第一个 episode 内 |
| `term_cause` | `[N]` | int8 | 0=time_out, 1=illegal_contact, 2=terrain_out_of_bounds, 3=bad_orientation, **4=reached_goal**, -1=loop 结束仍未触发任何终止（不应该出现，episode_length=10s 保证 time_out 兜底） |
| `term_step` | `[N]` | int32 | 第几步死的（-1=未死，否则 0..T-1） |
| `root_pos_xy` | `[N, T, 2]` | float32 | world xy，用于距离计算 |
| `cmd` | `[N, T, 3]` | float32 | velocity commands (vx, vy, wz) |
| `actual_vel` | `[N, T, 3]` | float32 | body-frame 实际速度 (vx, vy, wz) |
| `gate_leg` | `[N, T, num_leg]` | float16 | leg gate softmax 权重 |
| `gate_wheel` | `[N, T, num_wheel]` | float16 | wheel gate softmax 权重 |
| `gru_latent_sample` | `[N', T', latent_dim]` | float16 | **降采样**：N'=500 envs（按 sub-terrain 均匀采，每 sub-terrain 取 ~38 envs）× T'=50 步（每 10 步取一帧） |
| `reward_terms` | `[N, K]` | float32 | 每 env 第一个 episode 的各 reward term 平均（见 §2.5） |
| `reward_term_names` | `[K]` | str list | 对应名字（启动时探测 ep_infos 拿全集） |

**Gate 抓取**：复用 `play_moe.py:hook_fn` 模式 — `model.wheel_gate.register_forward_hook(...)` 抓 logits，再 softmax。注意只在 rollout (`model.eval()`) 时 hook 命中一次/step，无重复。

**GRU latent 抓取**：另在 `model.rnn` 上挂 forward_hook，取 output[0]（rnn_out），shape `(T_seq, B, latent_dim)`，取最后一帧 latent。

### 2.5 episode 边界处理

`base_env.scene.terrain.terrain_levels` / `terrain_types` 在 `curriculum=False` 时不会变（已在 §2.2 关掉 terrain_levels curriculum）。每个 env 只统计**第一个 episode**：

- 维护 `first_done[N]` bool，初始全 False
- 每 step 后读 `env.dones`（同 reset_buf），对 `dones[i] & not first_done[i]`：
  1. 从 `info["log"]` 抽 `Episode_Termination/*` 找出 True 的那个，map 到 enum 写入 `term_cause[i]`
  2. 抽 `Episode_Reward/*` 全部 key 写入 `reward_terms[i, :]`（启动时首次 done 探测拿全集，确定 `reward_term_names`、`K`）
  3. 写 `term_step[i] = t`
  4. 置 `first_done[i] = True`
- 每 step **写入** per-step buffer (root_pos / gate / etc.) 时，只对 `not first_done[i]` 的 env 写；已 first_done 的 env 后续步保留之前的零（plot 时按 `alive_mask` 截断）
- `alive_mask[i, t] = (term_step[i] == -1) | (t < term_step[i])`（plot 阶段 derive）
- 循环条件：`t < num_steps` AND `not all(first_done)`

**Termination key 探测**：第一次 done 触发时 dump `[k for k in info["log"] if k.startswith("Episode_Termination/")]`，建立 `name → enum int` 表，写到 `summary.json`。未知 key 退到 enum=255 并打 WARN。

**reward_term_names 探测**：第一次 done 时取 `[k.replace("Episode_Reward/", "") for k in info["log"] if k.startswith("Episode_Reward/")]`，长度即 K。后续 done 若 key 顺序不一致按名字 align。

### 2.6 距离/成功计算

由于 §2.2.1 已设 heading=0、`reached_goal` 直接在 sim 端按 +x 位移触发，成功判定退化为读 `term_cause` 即可：

```
success[i] = (term_cause[i] == reached_goal)
```

`time_out` 都算 fail（10s 内没走完 4m）。`success_dist` 由 sim 时 CLI `--success_dist` 决定（默认 4.0），写到 `summary.json`，plot 阶段只读不重算。

另外存一个 `max_disp_x[i] = max_t(root_pos_xy[i, t, 0] - env_origins[i, 0])` 供 boxplot/分布图诊断使用（看 fail 的 env 走了多远）。

## 3. 绘图实现（plot_moe_eval.py）

### 3.1 调用

```bash
python scripts/reinforcement_learning/rsl_rl/plot_moe_eval.py \
    --data_dir logs/moe_eval/<run_name>/<timestamp> \
    [--success_dist 4.0]
```

### 3.2 9 张图清单

| # | 文件 | 类型 | 内容 |
|---|------|------|------|
| 1 | `success_heatmap.png` | imshow | 30 行 level × 13 列 sub-terrain（terrain_generator 18 列里的 13 种 unique sub-terrain，重复列按 sub-terrain name 合并），颜色=成功率，cell 内 annot 样本数 |
| 2 | `expert_activation_bars.png` | 两子图堆叠条 | 上：leg expert 平均权重 by sub-terrain；下：wheel expert by sub-terrain |
| 3 | `velocity_tracking_box.png` | 3 子图 boxplot | vx_err / vy_err / wz_err 各按 sub-terrain 分组 |
| 4 | `termination_reward_breakdown.png` | 两子图 | 上：termination cause 比例堆叠条 by sub-terrain；下：top-8 reward terms 平均值 heatmap |
| 5 | `leg_wheel_coactivation.png` | imshow | `(num_leg × num_wheel)` 联合权重矩阵，全局聚合（或按 sub-terrain 分面） |
| 6 | `gate_entropy_per_terrain.png` | 双子图 bar | leg / wheel entropy by sub-terrain，水平线=`log(num_expert)` 上界 |
| 7 | `expert_switching_freq.png` | bar | 每 sub-terrain 平均 expert 切换次数/episode（leg + wheel 两组） |
| 8 | `gru_latent_tsne.png` | scatter | sklearn t-SNE 降到 2D，13 种 sub-terrain 13 种颜色 |
| 9 | `survival_curve.png` | line | 13 条曲线，alive 比例 vs step |

**主色方案**：matplotlib `tab20` (13 种 sub-terrain 用其中 13 色)、`viridis` (heatmap)。

**Expert 切换定义**：`dominant_expert[i, t] = argmax(gate_*[i, t])`；switching count = `sum_t (dominant[i, t] != dominant[i, t-1]) * alive_mask[i, t]`。

**Entropy**：`H = -sum(p * log(p + 1e-8))`，平均 over `(env, step)`，再按 sub-terrain 聚合。

**t-SNE**：原始 buffer 500 envs × 50 frames × 256 dim；为图清晰度，对每个 env 取**最后一帧 alive 的 latent**（500 个点），sklearn `TSNE(perplexity=30, n_iter=1000)` 降到 2D，按 sub-terrain 染色。预计 ~5-10s 计算。

### 3.3 依赖

- `matplotlib >= 3.5`
- `numpy`
- `scikit-learn`（仅 t-SNE 用，import 失败时跳过图 8 并打 WARN）

写入项目 `pyproject.toml` 的 dev deps 检查（不强制加，但脚本 import 失败时给清晰 warning）。

## 4. 输出目录布局

```
logs/moe_eval/
└── split_moe_teacher_parallel/
    └── 2026-05-18_15-30-00_iter19999/
        ├── raw.npz                        # 所有 buffer
        ├── summary.json                   # 命令、num_envs、success_dist、ckpt path、experts info、term cause label map
        ├── reward_term_summary.csv        # 表格便于 paper 引用
        └── plots/
            ├── 01_success_heatmap.png
            ├── 02_expert_activation_bars.png
            ├── 03_velocity_tracking_box.png
            ├── 04_termination_reward_breakdown.png
            ├── 05_leg_wheel_coactivation.png
            ├── 06_gate_entropy_per_terrain.png
            ├── 07_expert_switching_freq.png
            ├── 08_gru_latent_tsne.png
            └── 09_survival_curve.png
```

## 5. 已识别的风险 / 边界情况

| 风险 | 缓解 |
|------|------|
| `MOE_ROUGH_TERRAINS_CFG` 18 列中 13 种 sub-terrain（部分有重复列）→ 聚合时按 `terrain_generator.sub_terrains.keys()` 的 index | 通过 `terrain_types` 整数索引到 sub-terrain name list，重复列直接合并 |
| ep_info key 名字与 IsaacLab 版本相关 | 第一步先 dump 一次 `info["log"].keys()`，按实际 key 兜底处理 |
| GRU latent buffer 内存（500×50×256×fp16 ≈ 13MB） | 已按降采样设计 |
| `--strict_per_terrain` 13 次重启 sim | 仅 opt-in，默认不走 |
| 若策略 ckpt 是 distilled (`student.*` 前缀) | 复用 `train_moe.py:546` 的 strip 逻辑 |
| 全关 events 后 robot 启动 z=0 可能穿模 | 保留 `randomize_reset_base.pose_range.z = (0, 0)` 用 terrain spawn 默认 z（IsaacLab 在 spawn 时按 terrain origin + 默认高度自动抬升） |
| `reached_goal` 用 `env.scene.env_origins[:, 0]` 作为原点，curriculum=False 下 env_origins 在 reset 之间不变 | OK；如果做 strict 路径单 sub-terrain 评估，每 sub-terrain 重启时 env_origins 也是新的 |
| heading_command=True + heading=(0,0) + ang_vel_z=(0,0)，机器人自身 yaw 任何小漂移都会让 heading 控制器算出非零 ang_vel_cmd | 这是预期行为：heading 控制器自动算 ang_vel 把 yaw 拉回 0，所以 cmd_ang_vel 不一定为 0；boxplot 的 wz_err 因此还是有信号 |

## 6. 验证策略

1. **冒烟测试**：先用 `--num_envs 200 --num_steps 100` 跑一次 ~30s，确认所有 buffer 形状正确、`raw.npz` 可读、9 张图都生成
2. **主跑**：默认参数（2000 envs × 500 step）跑一次完整 eval，目视检查图样
3. **数值 sanity**：
   - 成功率 heatmap 在 level=0（最低难度）应≈100%（reached_goal）；level=29 应明显下降
   - flat / random_rough 列上 level 0-10 应 ≈ 100% success
   - leg expert 权重在 stairs 上应与 flat 上不同
   - gate entropy 应 < `log(num_leg_experts)`
   - vx 误差中位数应 < 0.5 m/s（cmd=1.0，期望 actual_vx ≈ 1.0）
   - max_disp_x 在 success env 上应 ≥ 4.0；在 fail env 上应 < 4.0

## 7. 范围外

- 不做 video 录制（已有 `play_moe.py --video` 流程）
- 不做 student 策略 distillation 评估（独立 task）
- 不做 跨多个 ckpt iter 的对比（可后续加 `compare_moe_eval.py`，本 spec 不含）
- 不写单元测试（脚本性质，靠冒烟测试验证）
