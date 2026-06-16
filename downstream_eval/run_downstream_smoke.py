# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rigorous smoke test that each downstream task ACTUALLY works.

For every task we run a **positive** case (a system that should work on real signal) and a
**control** case (signal destroyed: shuffled references / random labels / untrained policy).
A task "actually works" only if the positive case clearly beats its control — proving the
runner responds to real structure rather than always returning the same number.

Run::

    python -m downstream_eval.run_downstream_smoke
"""

import numpy as np

from downstream_eval.common.types import ClipRecord, WindowRecord
from downstream_eval.downstream.action_recognition_runner import run_linear_probe, run_nearest_centroid
from downstream_eval.downstream.encoders import HashingEncoder
from downstream_eval.downstream.policy_learning.synthetic import (
    RandomPolicy,
    make_tasks,
    rollout_policy_on_tasks,
    train_bc_policy,
)
from downstream_eval.downstream.retrieval_runner import run_caption_retrieval
from downstream_eval.downstream.robot_completion import robot_task_metrics


def _ok(label: str, *, passed: bool, detail: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")
    return passed


def test_retrieval_actually_works() -> bool:
    """Correct caption->clip retrieval should succeed; shuffled references should not."""
    print("\n== Retrieval ==")
    captions = [
        "a robot arm picks up the red block from the table",
        "a person chops the onions on a wooden cutting board",
        "the gripper places the blue cup onto the shelf",
        "a hand pours water from a kettle into a mug",
        "the arm pushes the green button on the panel",
        "someone stirs the soup in the metal pot",
    ]
    clips = [
        ClipRecord(uuid=f"c{i}", source_video="v.mp4", span=(0.0, 1.0),
                   windows=[WindowRecord(0, 30, {"qwen": cap})])
        for i, cap in enumerate(captions)
    ]
    # Paraphrased GT instruction per clip (lexical overlap but not identical).
    paraphrases = {
        "c0": "robot grabs the red block off the table",
        "c1": "person cuts onions on a cutting board",
        "c2": "gripper sets the blue cup on the shelf",
        "c3": "hand pours water into a mug from a kettle",
        "c4": "arm presses the green panel button",
        "c5": "someone stirs soup in a pot",
    }
    enc = HashingEncoder(dim=512)
    positive = run_caption_retrieval(clips, paraphrases, encoder=enc, ks=(1, 3)).metrics

    # Control: same texts, but mapped to the WRONG clips (signal destroyed).
    shuffled_ids = ["c1", "c2", "c3", "c4", "c5", "c0"]
    shuffled = {cid: paraphrases[orig] for cid, orig in zip([c.uuid for c in clips], shuffled_ids, strict=True)}
    control = run_caption_retrieval(clips, shuffled, encoder=enc, ks=(1, 3)).metrics

    return _ok(
        "correct refs retrieve their clip, shuffled refs do not",
        passed=positive["recall@1"] >= 0.8 and positive["recall@1"] - control["recall@1"] >= 0.5,
        detail=f"recall@1 positive={positive['recall@1']:.2f} vs control={control['recall@1']:.2f}",
    )


def _class_embeddings(seed: int, num_per_class: int = 8) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    centers = np.array([[4, 0], [0, 4], [-4, 0], [0, -4]], dtype=np.float32)
    embs, labels = [], []
    for cls, center in enumerate(centers):
        for _ in range(num_per_class):
            embs.append(center + 0.3 * rng.standard_normal(2).astype(np.float32))
            labels.append(cls)
    return np.stack(embs), np.array(labels, dtype=np.int64), ["reach", "grasp", "place", "push"]


def test_action_recognition_actually_works() -> bool:
    """Classifiers should learn separable classes; random labels should give ~chance."""
    print("\n== Action recognition ==")
    embs, labels, names = _class_embeddings(seed=0)
    chance = 1.0 / len(names)

    centroid = run_nearest_centroid(embs, labels, names, train_frac=0.6, seed=0).metrics["top1_accuracy"]
    probe = run_linear_probe(embs, labels, names, train_frac=0.6, seed=0).metrics["top1_accuracy"]

    rng = np.random.default_rng(1)
    random_labels = rng.integers(0, len(names), size=len(labels)).astype(np.int64)
    control = run_linear_probe(embs, random_labels, names, train_frac=0.6, seed=0).metrics["top1_accuracy"]

    return _ok(
        "centroid + linear probe learn separable classes; random labels ~chance",
        passed=centroid >= 0.9 and probe >= 0.9 and control <= chance + 0.2,
        detail=f"centroid={centroid:.2f} probe={probe:.2f} control(random)={control:.2f} chance={chance:.2f}",
    )


def test_policy_learning_actually_works() -> bool:
    """A behaviour-cloned policy should solve held-out tasks; a random policy should not."""
    print("\n== Policy learning (behaviour cloning) ==")
    all_tasks = make_tasks(40, seed=7)
    train_tasks, test_tasks = all_tasks[:30], all_tasks[30:]

    bc_policy = train_bc_policy(train_tasks)  # actually fits a regressor on expert demos
    learned = robot_task_metrics(rollout_policy_on_tasks(bc_policy, test_tasks, seed=0))
    control = robot_task_metrics(rollout_policy_on_tasks(RandomPolicy(np.random.default_rng(0)), test_tasks, seed=0))

    return _ok(
        "trained BC policy succeeds on held-out tasks; random policy fails",
        passed=learned["task_success_rate"] >= 0.8
        and learned["task_success_rate"] - control["task_success_rate"] >= 0.5,
        detail=(
            f"success learned={learned['task_success_rate']:.2f} vs random={control['task_success_rate']:.2f}, "
            f"learned efficiency={learned['efficiency_ratio']:.2f}"
        ),
    )


def main() -> None:
    """Run all task-level smoke checks and exit non-zero on any failure."""
    results = {
        "retrieval": test_retrieval_actually_works(),
        "action_recognition": test_action_recognition_actually_works(),
        "policy_learning": test_policy_learning_actually_works(),
    }
    print("\n== Summary ==")
    for task, passed in results.items():
        print(f"  {task}: {'WORKS' if passed else 'BROKEN'}")
    if not all(results.values()):
        raise SystemExit(1)
    print("\nAll downstream tasks actually work.")


if __name__ == "__main__":
    main()
