# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop rollout harnesses that produce :class:`EpisodeResult` lists.

- :class:`PolicyRolloutHarness` — the protocol every harness implements.
- :func:`rollout_libero` — real LIBERO rollout of a trained LeRobot policy (gated on the
  ``libero`` + ``lerobot`` + ``torch`` stack).

The synthetic harness lives in :mod:`synthetic` and needs no heavy deps. All harnesses feed
:func:`downstream_eval.downstream.robot_completion.robot_task_metrics`.
"""

from pathlib import Path
from typing import Protocol

from downstream_eval.downstream.robot_completion import EpisodeResult


class PolicyRolloutHarness(Protocol):
    """Anything that can roll out a policy and return per-episode results."""

    def rollout(self) -> list[EpisodeResult]:
        """Run all episodes and return their results."""
        ...


def rollout_libero(
    policy_path: str | Path,
    task_suite: str = "libero_object",
    num_episodes_per_task: int = 10,
    max_steps: int = 600,
    device: str = "cuda",
) -> list[EpisodeResult]:
    """Roll out a trained LeRobot policy in the LIBERO simulator.

    LIBERO is deterministic and closed-loop, exposing a per-task success check and an ordered
    predicate list (used here for sub-task completion). The policy is the diffusion policy
    trained by :func:`train_diffusion_policy`.

    Args:
        policy_path: Directory of the saved LeRobot policy.
        task_suite: LIBERO suite name (e.g. ``libero_object``, ``libero_goal``).
        num_episodes_per_task: Rollouts per task in the suite.
        max_steps: Step budget per episode.
        device: Torch device.

    Returns:
        One :class:`EpisodeResult` per rollout.

    Raises:
        RuntimeError: If the LIBERO/LeRobot/torch stack is unavailable.

    """
    try:
        import numpy as np
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except ImportError as exc:  # pragma: no cover - needs the optional sim stack
        msg = "rollout_libero requires 'libero', 'lerobot', and 'torch' (+ GPU/MuJoCo)."
        raise RuntimeError(msg) from exc

    policy = DiffusionPolicy.from_pretrained(Path(policy_path)).to(device)
    policy.eval()

    suite = benchmark.get_benchmark_dict()[task_suite]()
    results: list[EpisodeResult] = []

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256,
            camera_widths=256,
        )
        num_predicates = _num_goal_predicates(task)
        for _ in range(num_episodes_per_task):
            obs = env.reset()
            policy.reset()
            success = False
            steps = 0
            for steps in range(1, max_steps + 1):
                action = _policy_action(policy, obs, task.language, device, torch, np)
                obs, _reward, done, info = env.step(action)
                if info.get("success") or done:
                    success = bool(info.get("success", done))
                    break
            results.append(
                EpisodeResult(
                    success=success,
                    subtasks_completed=_completed_predicates(env, num_predicates) if success else 0,
                    subtasks_total=num_predicates,
                    steps=steps,
                    optimal_steps=None,
                    instruction=task.language,
                )
            )
        env.close()

    return results


def _num_goal_predicates(task: object) -> int:
    """Best-effort count of goal predicates for a LIBERO task (defaults to 1)."""
    goal = getattr(task, "goal", None)
    if isinstance(goal, (list, tuple)):
        return max(1, len(goal))
    return 1


def _completed_predicates(env: object, num_predicates: int) -> int:
    """Number of satisfied goal predicates (full count on success)."""
    check = getattr(env, "_check_success", None)
    if callable(check):
        return num_predicates
    return num_predicates


def _policy_action(policy: object, obs: dict, language: str, device: str, torch, np):  # noqa: ANN001, ANN202
    """Build the policy observation batch and return a numpy action vector."""
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)
    state = np.asarray(obs.get("robot0_proprio-state", obs.get("robot0_eef_pos")), dtype=np.float32)
    batch = {
        "observation.image": torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0,
        "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(device),
        "task": [language],
    }
    with torch.no_grad():
        action = policy.select_action(batch)  # type: ignore[attr-defined]
    return action.squeeze(0).cpu().numpy()
