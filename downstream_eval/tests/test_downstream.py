# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for downstream metrics: retrieval, action recognition, robot completion."""

import math

import numpy as np

from downstream_eval.downstream.action_recognition import (
    evaluate_action_recognition,
    macro_f1,
    mean_per_class_accuracy,
    topk_accuracy,
)
from downstream_eval.downstream.retrieval import evaluate_retrieval, retrieval_metrics
from downstream_eval.downstream.robot_completion import EpisodeResult, robot_task_metrics


def test_retrieval_perfect() -> None:
    sim = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.2, 0.7]], dtype=np.float32)
    out = retrieval_metrics(sim, [[0], [1], [2]], ks=(1, 2))
    assert out["recall@1"] == 1.0
    assert out["mrr"] == 1.0
    assert out["ndcg@1"] == 1.0


def test_retrieval_correct_at_rank_two() -> None:
    sim = np.array([[0.2, 0.9]], dtype=np.float32)  # correct index 0 is ranked 2nd
    out = retrieval_metrics(sim, [[0]], ks=(1, 2))
    assert out["recall@1"] == 0.0
    assert out["recall@2"] == 1.0
    assert math.isclose(out["mrr"], 0.5)
    assert math.isclose(out["ndcg@2"], 1.0 / math.log2(3), rel_tol=1e-6)


def test_retrieval_both_directions_identity() -> None:
    emb = np.eye(4, dtype=np.float32)
    out = evaluate_retrieval(emb, emb, ks=(1,))
    assert out["text_to_clip"]["recall@1"] == 1.0
    assert out["clip_to_text"]["recall@1"] == 1.0


def test_action_recognition_perfect() -> None:
    scores = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    out = evaluate_action_recognition(np.array([0, 1]), scores=scores, num_classes=2, ks=(1,))
    assert out["top1_accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["mean_per_class_accuracy"] == 1.0


def test_topk_and_macro_f1_imbalance() -> None:
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])  # never predicts minority class
    assert mean_per_class_accuracy(y_true, y_pred, 2) == 0.5
    # class0 f1 = 2*0.75*1/(1.75)=0.857; class1 f1 = 0 -> macro ~0.4286
    assert math.isclose(macro_f1(y_true, y_pred, 2), (2 * 0.75 / 1.75) / 2, rel_tol=1e-6)


def test_topk_accuracy_k_clamped() -> None:
    scores = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    assert topk_accuracy(scores, np.array([0]), k=5) == 1.0  # k clamped to num classes


def test_robot_task_metrics() -> None:
    episodes = [
        EpisodeResult(success=True, subtasks_completed=4, subtasks_total=4, steps=100, optimal_steps=100),
        EpisodeResult(success=False, subtasks_completed=2, subtasks_total=4, steps=200, optimal_steps=100),
    ]
    out = robot_task_metrics(episodes)
    assert out["task_success_rate"] == 0.5
    assert out["subtask_completion_rate"] == 0.75  # (4/4 + 2/4)/2
    assert out["efficiency_ratio"] == 1.0  # only successful episode, 100/100


def test_robot_metrics_empty() -> None:
    out = robot_task_metrics([])
    assert math.isnan(out["task_success_rate"])
    assert out["num_episodes"] == 0.0
