# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that the downstream *task runners* actually perform their task end-to-end."""

import numpy as np

from downstream_eval.common.types import ClipRecord, WindowRecord
from downstream_eval.downstream.action_recognition_runner import (
    run_linear_probe,
    run_nearest_centroid,
    run_zero_shot_language,
)
from downstream_eval.downstream.encoders import HashingEncoder, build_text_encoder
from downstream_eval.downstream.policy_learning.synthetic import (
    NoisyPolicy,
    RandomPolicy,
    ScriptedExpert,
    make_tasks,
    rollout_policy_on_tasks,
    run_synthetic_rollouts,
    train_bc_policy,
)
from downstream_eval.downstream.retrieval_runner import run_caption_retrieval, run_text_to_clip_retrieval
from downstream_eval.downstream.robot_completion import robot_task_metrics


def _clip(uuid: str, caption: str) -> ClipRecord:
    return ClipRecord(
        uuid=uuid,
        source_video="v.mp4",
        span=(0.0, 1.0),
        windows=[WindowRecord(start_frame=0, end_frame=30, captions={"qwen": caption})],
    )


def test_hashing_encoder_is_deterministic_and_normalised() -> None:
    enc = HashingEncoder(dim=64)
    a = enc.encode(["pick up the red block"])
    b = enc.encode(["pick up the red block"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_build_text_encoder_auto_falls_back_to_hashing() -> None:
    enc = build_text_encoder("auto")
    assert enc.name.startswith(("hashing", "sbert"))


def test_caption_retrieval_recovers_matching_clip() -> None:
    clips = [
        _clip("c1", "a robot picks up the red block"),
        _clip("c2", "a person chops the onions"),
        _clip("c3", "the arm places the cup on the shelf"),
    ]
    references = {
        "c1": "robot picks up the red block",
        "c2": "person chops onions",
        "c3": "arm places cup on shelf",
    }
    run = run_caption_retrieval(clips, references, encoder=HashingEncoder(dim=128), ks=(1, 2))
    assert run.metrics["recall@1"] == 1.0
    assert run.top_k(0, 1) == ["c1"]


def test_text_to_clip_retrieval_in_shared_space() -> None:
    clip_embeddings = {
        "c1": np.array([1.0, 0.0], dtype=np.float32),
        "c2": np.array([0.0, 1.0], dtype=np.float32),
    }

    class _Stub:
        name = "stub"

        def encode(self, texts: list[str]) -> np.ndarray:
            table = {"left": [1.0, 0.0], "right": [0.0, 1.0]}
            return np.array([table[t] for t in texts], dtype=np.float32)

    run = run_text_to_clip_retrieval(
        ["c1", "c2"], clip_embeddings, {"left": ["c1"], "right": ["c2"]}, encoder=_Stub(), ks=(1,)
    )
    assert run.metrics["recall@1"] == 1.0


def _separable_embeddings(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    centers = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0]], dtype=np.float32)
    embs = []
    labels = []
    for cls, center in enumerate(centers):
        for _ in range(6):
            embs.append(center + 0.2 * rng.standard_normal(2).astype(np.float32))
            labels.append(cls)
    return np.stack(embs), np.array(labels, dtype=np.int64), ["reach", "grasp", "place"]


def test_nearest_centroid_runs_and_classifies() -> None:
    embs, labels, names = _separable_embeddings()
    run = run_nearest_centroid(embs, labels, names, train_frac=0.5, seed=0)
    assert run.method == "nearest_centroid"
    assert run.metrics["top1_accuracy"] >= 0.9
    assert len(run.predictions) == len(run.test_indices)


def test_linear_probe_runs_and_classifies() -> None:
    embs, labels, names = _separable_embeddings()
    run = run_linear_probe(embs, labels, names, train_frac=0.5, seed=0)
    assert run.metrics["top1_accuracy"] >= 0.9


def test_zero_shot_language_uses_captions() -> None:
    captions = ["reach toward the object", "grasp the handle", "place it down gently"]
    labels = np.array([0, 1, 2], dtype=np.int64)
    run = run_zero_shot_language(captions, labels, ["reach", "grasp", "place"], encoder=HashingEncoder(dim=256))
    assert 0.0 <= run.metrics["top1_accuracy"] <= 1.0
    assert len(run.predictions) == 3


def test_synthetic_expert_always_succeeds() -> None:
    rollouts = run_synthetic_rollouts(lambda _rng: ScriptedExpert(), num_episodes=8, seed=0)
    metrics = robot_task_metrics(rollouts)
    assert metrics["task_success_rate"] == 1.0
    assert metrics["efficiency_ratio"] <= 1.0
    assert metrics["num_episodes"] == 8.0


def test_synthetic_noisy_policy_is_not_better_than_expert() -> None:
    expert = robot_task_metrics(run_synthetic_rollouts(lambda _rng: ScriptedExpert(), num_episodes=8, seed=3))
    noisy = robot_task_metrics(
        run_synthetic_rollouts(lambda rng: NoisyPolicy(noise=1.2, rng=rng), num_episodes=8, seed=3)
    )
    assert noisy["task_success_rate"] <= expert["task_success_rate"]


def test_behavior_cloning_learns_and_generalizes() -> None:
    """A BC policy trained on expert demos solves held-out tasks; random policy does not."""
    tasks = make_tasks(40, seed=7)
    train_tasks, test_tasks = tasks[:30], tasks[30:]
    bc_policy = train_bc_policy(train_tasks)

    learned = robot_task_metrics(rollout_policy_on_tasks(bc_policy, test_tasks, seed=0))
    random = robot_task_metrics(
        rollout_policy_on_tasks(RandomPolicy(np.random.default_rng(0)), test_tasks, seed=0)
    )
    assert learned["task_success_rate"] >= 0.8
    assert learned["task_success_rate"] - random["task_success_rate"] >= 0.5
