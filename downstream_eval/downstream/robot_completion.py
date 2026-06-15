# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot task-completion metrics for closed-loop policy evaluation.

This module scores *rollout results* produced by an external simulator/policy stack
(LIBERO sim, LeRobot, or a real-robot harness). It deliberately does **not** train or
run any policy itself — that requires heavy, environment-specific infrastructure. Instead
it defines a stable :class:`EpisodeResult` schema and a :class:`RolloutProvider` protocol,
plus a JSON loader, so a rollout harness can feed results in and get comparable numbers.

Metrics
-------
- ``task_success_rate``      : fraction of episodes that reached the goal.
- ``subtask_completion_rate``: mean fraction of sub-tasks completed per episode.
- ``efficiency_ratio``       : mean ``optimal_steps / steps`` over successful episodes (<=1, higher better).

See ``downstream_eval/downstream/README.md`` for how to wire LIBERO / LeRobot.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class EpisodeResult:
    """Outcome of a single closed-loop rollout episode.

    Attributes:
        success: Whether the full task goal was reached.
        subtasks_completed: Number of sub-tasks completed.
        subtasks_total: Number of sub-tasks defined for the task.
        steps: Environment steps taken by the policy.
        optimal_steps: Reference/optimal step count (e.g. demo length), if known.
        instruction: Optional language instruction used to condition the policy.

    """

    success: bool
    subtasks_completed: int = 0
    subtasks_total: int = 0
    steps: int = 0
    optimal_steps: int | None = None
    instruction: str | None = None


class RolloutProvider(Protocol):
    """Anything that can produce closed-loop rollout results for evaluation."""

    def rollout(self) -> list[EpisodeResult]:
        """Run rollouts and return their per-episode results."""
        ...


def robot_task_metrics(episodes: list[EpisodeResult]) -> dict[str, float]:
    """Aggregate per-episode rollout results into task-completion metrics."""
    if not episodes:
        return {
            "task_success_rate": float("nan"),
            "subtask_completion_rate": float("nan"),
            "efficiency_ratio": float("nan"),
            "num_episodes": 0.0,
        }

    success_rate = sum(1 for e in episodes if e.success) / len(episodes)

    subtask_fracs = [
        (e.subtasks_completed / e.subtasks_total) for e in episodes if e.subtasks_total > 0
    ]
    subtask_rate = sum(subtask_fracs) / len(subtask_fracs) if subtask_fracs else float("nan")

    eff = [
        min(1.0, e.optimal_steps / e.steps)
        for e in episodes
        if e.success and e.optimal_steps is not None and e.steps > 0
    ]
    efficiency = sum(eff) / len(eff) if eff else float("nan")

    return {
        "task_success_rate": success_rate,
        "subtask_completion_rate": subtask_rate,
        "efficiency_ratio": efficiency,
        "num_episodes": float(len(episodes)),
    }


def load_rollouts_from_json(path: str | Path) -> list[EpisodeResult]:
    """Load rollout results from a JSON list of episode dicts.

    Expected schema per item::

        {"success": bool, "subtasks_completed": int, "subtasks_total": int,
         "steps": int, "optimal_steps": int | null, "instruction": str | null}
    """
    data = json.loads(Path(path).read_text())
    return [
        EpisodeResult(
            success=bool(item["success"]),
            subtasks_completed=int(item.get("subtasks_completed", 0)),
            subtasks_total=int(item.get("subtasks_total", 0)),
            steps=int(item.get("steps", 0)),
            optimal_steps=item.get("optimal_steps"),
            instruction=item.get("instruction"),
        )
        for item in data
    ]
