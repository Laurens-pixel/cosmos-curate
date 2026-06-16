<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Language-conditioned policy learning

The gold-standard downstream task: prove the curated data is *useful for learning a policy*.
We **perform** the task end-to-end and then score it — we do not just score precomputed
numbers.

```
curated (clip + caption)  ─┐
                           ├─►  LeRobotDataset  ─►  train diffusion policy  ─►  rollout  ─►  metrics
source dataset (actions,   │     (lerobot_adapter)      (train.py)          (rollout.py)  (robot_completion)
 proprio state) by ep id  ─┘
```

## Why LeRobot (open-source, directly applicable)
[LeRobot](https://github.com/huggingface/lerobot) is the repo that directly fits our setup:
it provides the dataset format, a **language-conditioned diffusion policy**, and training
utilities, and it accepts exactly the `(observation.image, observation.state, action, task)`
structure our curation produces once joined with the source dataset's action/state streams.
We therefore build a thin adapter on top of LeRobot rather than reimplementing a diffusion
policy. [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) supplies the deterministic
closed-loop simulator for evaluation.

## Files
| File | Role | Requires |
| --- | --- | --- |
| `lerobot_adapter.py` | `CuratedEpisode` → `LeRobotDataset` on disk | `lerobot` |
| `train.py` | train `DiffusionPolicy` on that dataset | `lerobot` + `torch` + GPU |
| `rollout.py` | `rollout_libero(...)` → `list[EpisodeResult]` | `libero` + `lerobot` + `torch` + MuJoCo |
| `synthetic.py` | pure-numpy reach env + policies | nothing (always runnable) |

The heavy modules use lazy imports and raise a clear `RuntimeError` if their dependency is
missing, so the rest of the suite (and CI) runs without them.

## Real run (LIBERO ★ — closed-loop)
```python
from downstream_eval.downstream.policy_learning.lerobot_adapter import CuratedEpisode, build_lerobot_dataset
from downstream_eval.downstream.policy_learning.train import TrainConfig, train_diffusion_policy
from downstream_eval.downstream.policy_learning.rollout import rollout_libero
from downstream_eval.downstream.robot_completion import robot_task_metrics

# 1. Join curated clips/captions with the source dataset's actions+state into episodes.
episodes = [CuratedEpisode(frames=..., actions=..., states=..., instruction=caption)]
build_lerobot_dataset(episodes, repo_id="local/curated_libero", root="/data/curated_libero")

# 2. Train a language-conditioned diffusion policy.
ckpt = train_diffusion_policy(TrainConfig(
    dataset_repo_id="local/curated_libero", dataset_root="/data/curated_libero", output_dir="/ckpt"))

# 3. Roll out closed-loop in LIBERO and score.
results = rollout_libero(ckpt, task_suite="libero_object", num_episodes_per_task=20)
print(robot_task_metrics(results))   # task_success_rate, subtask_completion_rate, efficiency_ratio
```

### Which datasets
- **LIBERO ★** — deterministic sim, closed-loop, released task suites → primary target.
- **AgiBotWorld / DROID** — have actions+state but evaluation is real-robot (or needs a learned
  simulator); use the same adapter + training, then evaluate on whatever rollout harness you
  have and feed `EpisodeResult`s in.

## Smoke / plumbing test (no GPU, no sim)
`synthetic.py` is a 2-D multi-waypoint reach env that exercises the exact
rollout → `EpisodeResult` → `robot_task_metrics` path:

```python
from downstream_eval.downstream.policy_learning.synthetic import run_synthetic_rollouts, ScriptedExpert, NoisyPolicy
from downstream_eval.downstream.robot_completion import robot_task_metrics

robot_task_metrics(run_synthetic_rollouts(lambda rng: ScriptedExpert()))            # success 1.0, efficiency 1.0
robot_task_metrics(run_synthetic_rollouts(lambda rng: NoisyPolicy(0.9, rng)))       # lower efficiency
```

To plug a *real* LeRobot policy into the synthetic env (or any custom env), implement the
`Policy.act(pos, task, next_wp)` interface, or implement the `PolicyRolloutHarness` protocol
in `rollout.py` for your own simulator.
