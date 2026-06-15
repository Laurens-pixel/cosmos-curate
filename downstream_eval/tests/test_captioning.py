# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for caption-quality metrics."""

import numpy as np

from downstream_eval.captioning.clipscore import clipscore_from_embeddings
from downstream_eval.captioning.judge_aggregate import aggregate_judgments
from downstream_eval.captioning.reference_metrics import bertscore, cider_d, meteor
from downstream_eval.captioning.temporal import (
    action_order_consistency,
    lexical_coherence,
    temporal_grounding_accuracy,
)


def test_cider_identical_high_disjoint_low() -> None:
    high = cider_d(["the robot picks the block"], [["the robot picks the block"]])["cider_d"]
    low = cider_d(["completely unrelated words here"], [["the robot picks the block"]])["cider_d"]
    assert high > low
    assert low == 0.0


def test_meteor_identical_and_disjoint() -> None:
    assert meteor(["pick up the block"], [["pick up the block"]])["meteor"] > 0.9
    assert meteor(["xyz qrs"], [["pick up the block"]])["meteor"] == 0.0


def test_bertscore_graceful_when_missing() -> None:
    out = bertscore(["a"], [["a"]])
    assert "available" in out  # never raises regardless of install state


def test_clipscore_aligned_vs_orthogonal() -> None:
    img = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    aligned = clipscore_from_embeddings(img, img)
    orthogonal = clipscore_from_embeddings(img, img[::-1])
    assert np.isclose(aligned["clipscore"], 2.5)
    assert orthogonal["clipscore"] == 0.0


def test_lexical_coherence_range() -> None:
    score = lexical_coherence(["the robot grasps the block", "the robot lifts the block"])
    assert 0.0 < score <= 1.0


def test_temporal_grounding_accuracy() -> None:
    windows = [(0.0, 1.0, "chop the onions"), (1.0, 2.0, "stir the mixture")]
    intervals = [(0.0, 1.0, "chop the onions"), (1.0, 2.0, "stir the mixture")]
    assert temporal_grounding_accuracy(windows, intervals)["temporal_grounding_accuracy"] == 1.0


def test_action_order_consistency() -> None:
    assert action_order_consistency(["a", "b", "c"], ["a", "b", "c"])["action_order_kendall_tau"] == 1.0


def test_judge_aggregation_counts_and_agreement() -> None:
    records = [
        {"variant": "v", "verdict": "CORRECT", "score": 1, "gt_extras": {"human_correct": True}},
        {"variant": "v", "verdict": "INCORRECT", "score": 0, "gt_extras": {"human_correct": False}},
        {"variant": "v", "verdict": "CORRECT", "score": 1},
    ]
    agg = aggregate_judgments(records)["v"]
    assert agg["num"] == 3
    assert agg["verdict_distribution"]["CORRECT"] == 2
    assert agg["human_agreement"] == 1.0
    assert agg["f1_vs_human"] == 1.0
