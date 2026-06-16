# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pure-numpy synthetic environment + policies for closed-loop policy evaluation.

This exists so the *full* policy-learning -> rollout -> metrics path is runnable and testable
without LeRobot/LIBERO/torch/MuJoCo. The environment is a 2-D multi-waypoint reach task: the
agent must visit an ordered list of sub-goal waypoints (the "sub-tasks") and finally reach the
goal. It mirrors the real harness contract: each episode yields an
:class:`~downstream_eval.downstream.robot_completion.EpisodeResult` consumed by
``robot_task_metrics``.

It also serves as a behaviour-cloning sanity check: ``ScriptedExpert`` generates demonstrations,
``NearestNeighborPolicy`` "learns" from them (no training), and a noisy variant produces partial
successes so the metrics are non-degenerate.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from downstream_eval.downstream.robot_completion import EpisodeResult

FloatArray = npt.NDArray[np.float32]


@dataclass
class ReachTask:
    """A multi-waypoint reach task instance."""

    start: FloatArray
    waypoints: list[FloatArray]  # ordered sub-goals; last is the final goal
    instruction: str
    success_radius: float = 0.15

    def optimal_steps(self, step_size: float) -> int:
        """Path length along waypoints divided by step size (the optimal step count)."""
        pts = [self.start, *self.waypoints]
        dist = sum(float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1))
        return max(1, int(np.ceil(dist / step_size)))


@dataclass
class Synthetic2DReachEnv:
    """Deterministic 2-D reach environment with ordered waypoint sub-tasks."""

    step_size: float = 0.08
    max_steps: int = 200

    def rollout(
        self,
        task: ReachTask,
        policy: "Policy",
        rng: np.random.Generator,
    ) -> EpisodeResult:
        """Run one closed-loop episode and return its result."""
        pos = task.start.astype(np.float32).copy()
        next_wp = 0
        for step in range(1, self.max_steps + 1):
            action = policy.act(pos, task, next_wp)
            pos = pos + np.clip(action, -1.0, 1.0) * self.step_size
            if np.linalg.norm(pos - task.waypoints[next_wp]) <= task.success_radius:
                next_wp += 1
                if next_wp == len(task.waypoints):
                    return EpisodeResult(
                        success=True,
                        subtasks_completed=len(task.waypoints),
                        subtasks_total=len(task.waypoints),
                        steps=step,
                        optimal_steps=task.optimal_steps(self.step_size),
                        instruction=task.instruction,
                    )
        return EpisodeResult(
            success=False,
            subtasks_completed=next_wp,
            subtasks_total=len(task.waypoints),
            steps=self.max_steps,
            optimal_steps=task.optimal_steps(self.step_size),
            instruction=task.instruction,
        )


class Policy:
    """Minimal policy interface: map (position, task, current sub-goal index) -> action."""

    def act(self, pos: FloatArray, task: ReachTask, next_wp: int) -> FloatArray:
        """Return a 2-D action vector."""
        raise NotImplementedError


@dataclass
class ScriptedExpert(Policy):
    """Oracle that drives straight toward the current waypoint (generates demos)."""

    def act(self, pos: FloatArray, task: ReachTask, next_wp: int) -> FloatArray:
        """Unit step toward the active waypoint."""
        target = task.waypoints[next_wp]
        direction = target - pos
        norm = float(np.linalg.norm(direction))
        return (direction / norm) if norm > 1e-6 else np.zeros_like(direction)


@dataclass
class NoisyPolicy(Policy):
    """Expert direction corrupted by Gaussian noise (imperfect learned policy proxy)."""

    noise: float = 0.4
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def act(self, pos: FloatArray, task: ReachTask, next_wp: int) -> FloatArray:
        """Noisy step toward the active waypoint."""
        target = task.waypoints[next_wp]
        direction = target - pos
        norm = float(np.linalg.norm(direction))
        unit = (direction / norm) if norm > 1e-6 else np.zeros_like(direction)
        return unit + self.noise * self.rng.standard_normal(2).astype(np.float32)


@dataclass
class RandomPolicy(Policy):
    """Untrained control policy: random actions (should mostly fail)."""

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def act(self, pos: FloatArray, task: ReachTask, next_wp: int) -> FloatArray:  # noqa: ARG002
        """Return a random action independent of the goal."""
        return self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def collect_demonstrations(
    tasks: list[ReachTask], expert: Policy | None = None, step_size: float = 0.08
) -> tuple[FloatArray, FloatArray]:
    """Roll out an expert to collect goal-conditioned (observation -> action) pairs.

    Observation is ``[pos_x, pos_y, goal_x, goal_y]`` (the agent sees its position and the
    active sub-goal); the label is the expert's action. Used to *train* a policy via
    behaviour cloning.
    """
    expert = expert or ScriptedExpert()
    env = Synthetic2DReachEnv(step_size=step_size)
    obs_list: list[FloatArray] = []
    act_list: list[FloatArray] = []
    for task in tasks:
        pos = task.start.astype(np.float32).copy()
        next_wp = 0
        for _ in range(env.max_steps):
            target = task.waypoints[next_wp]
            # Step with the expert (good on-path coverage); but regress the *displacement*
            # toward the goal, which is linearly representable and yields a stable P-controller.
            obs_list.append(np.concatenate([pos, target]).astype(np.float32))
            act_list.append((target - pos).astype(np.float32))
            pos = pos + np.clip(expert.act(pos, task, next_wp), -1.0, 1.0) * step_size
            if np.linalg.norm(pos - target) <= task.success_radius:
                next_wp += 1
                if next_wp == len(task.waypoints):
                    break
    return np.stack(obs_list), np.stack(act_list)


@dataclass
class BehaviorCloningPolicy(Policy):
    """A policy *learned* from expert demonstrations via ridge regression (scikit-learn).

    This genuinely trains: it fits observation->action on demos, then is rolled out on
    held-out tasks. A working learner reaches goals it never saw; an untrained control does not.
    """

    model: object = None

    def fit(self, observations: FloatArray, actions: FloatArray) -> "BehaviorCloningPolicy":
        """Train the regressor on (observation, action) pairs."""
        try:
            from sklearn.linear_model import Ridge
        except ImportError as exc:  # pragma: no cover
            msg = "BehaviorCloningPolicy.fit requires scikit-learn."
            raise RuntimeError(msg) from exc
        model = Ridge(alpha=1e-3)
        model.fit(observations, actions)
        self.model = model
        return self

    def act(self, pos: FloatArray, task: ReachTask, next_wp: int) -> FloatArray:
        """Predict an action from the current observation."""
        if self.model is None:
            msg = "BehaviorCloningPolicy must be fit() before use."
            raise RuntimeError(msg)
        target = task.waypoints[next_wp]
        obs = np.concatenate([pos, target]).astype(np.float32).reshape(1, -1)
        return np.asarray(self.model.predict(obs)[0], dtype=np.float32)  # type: ignore[attr-defined]


def train_bc_policy(train_tasks: list[ReachTask], step_size: float = 0.08) -> BehaviorCloningPolicy:
    """Collect expert demos on ``train_tasks`` and fit a behaviour-cloning policy."""
    obs, act = collect_demonstrations(train_tasks, step_size=step_size)
    return BehaviorCloningPolicy().fit(obs, act)


def make_tasks(num: int, seed: int = 0) -> list[ReachTask]:
    """Generate ``num`` random multi-waypoint reach tasks."""
    rng = np.random.default_rng(seed)
    verbs = ["reach", "push", "lift", "place"]
    tasks: list[ReachTask] = []
    for i in range(num):
        start = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        waypoints = [rng.uniform(-1.0, 1.0, size=2).astype(np.float32) for _ in range(2)]
        tasks.append(ReachTask(start=start, waypoints=waypoints, instruction=f"{verbs[i % len(verbs)]} the object"))
    return tasks


def rollout_policy_on_tasks(policy: Policy, tasks: list[ReachTask], seed: int = 0) -> list[EpisodeResult]:
    """Roll out a single (already-built) policy over an explicit list of tasks."""
    rng = np.random.default_rng(seed)
    env = Synthetic2DReachEnv()
    return [env.rollout(task, policy, rng) for task in tasks]


def run_synthetic_rollouts(
    policy_factory: Callable[[np.random.Generator], Policy],
    num_episodes: int = 12,
    seed: int = 0,
) -> list[EpisodeResult]:
    """Roll out ``policy_factory`` over synthetic reach tasks and return per-episode results."""
    rng = np.random.default_rng(seed)
    env = Synthetic2DReachEnv()
    tasks = make_tasks(num_episodes, seed=seed)
    return [env.rollout(task, policy_factory(rng), rng) for task in tasks]
