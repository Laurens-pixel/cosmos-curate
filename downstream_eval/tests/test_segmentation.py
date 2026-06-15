# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for segmentation-quality metrics with hand-computed expectations."""

import math

from downstream_eval.common.types import Segment
from downstream_eval.segmentation.metrics import (
    boundary_f1,
    boundary_mae,
    completeness,
    damerau_levenshtein,
    evaluate_segmentation,
    kendall_tau,
    ordering_metrics,
    segment_boundaries,
    temporal_iou,
)


def _segs(spans: list[tuple[float, float, str]]) -> list[Segment]:
    return [Segment(start=s, end=e, label=lbl) for s, e, lbl in spans]


GT = _segs([(0, 1, "a"), (1, 2, "b"), (2, 3, "c")])


def test_boundaries_exclude_global_endpoints() -> None:
    assert segment_boundaries(GT) == [1.0, 2.0]


def test_perfect_segmentation() -> None:
    result = evaluate_segmentation(GT, GT, tolerance=0.1)
    assert result["boundary_f1"] == 1.0
    assert result["boundary_mae_s"] == 0.0
    assert result["mean_gt_iou"] == 1.0
    assert result["over_segmentation_rate"] == 0.0
    assert result["under_segmentation_rate"] == 0.0
    assert result["fragmentation_index"] == 1.0
    assert result["kendall_tau"] == 1.0
    assert result["damerau_levenshtein"] == 0.0


def test_over_segmentation_adds_false_positive_boundary() -> None:
    pred = _segs([(0, 1, "a"), (1, 1.5, "b"), (1.5, 2, "b"), (2, 3, "c")])
    bf1 = boundary_f1(pred, GT, tolerance=0.1)
    assert bf1["recall"] == 1.0
    assert bf1["fp"] == 1.0  # the extra cut at 1.5
    comp = completeness(pred, GT, tolerance=0.1)
    assert math.isclose(comp["over_segmentation_rate"], 1 / 3)
    assert comp["under_segmentation_rate"] == 0.0
    assert comp["fragmentation_index"] > 1.0


def test_under_segmentation_misses_boundary() -> None:
    pred = _segs([(0, 2, "a"), (2, 3, "c")])
    bf1 = boundary_f1(pred, GT, tolerance=0.1)
    assert bf1["fn"] == 1.0  # boundary at 1.0 missed
    assert bf1["recall"] == 0.5


def test_boundary_tolerance_window() -> None:
    pred = _segs([(0, 1.3, "a"), (1.3, 2, "b"), (2, 3, "c")])
    assert boundary_f1(pred, GT, tolerance=0.2)["tp"] == 1.0  # only boundary near 2.0 matches
    assert boundary_f1(pred, GT, tolerance=0.5)["tp"] == 2.0


def test_boundary_mae() -> None:
    pred = _segs([(0, 1.1, "a"), (1.1, 2.2, "b"), (2.2, 3, "c")])
    assert math.isclose(boundary_mae(pred, GT), (0.1 + 0.2) / 2, rel_tol=1e-6)


def test_temporal_iou_partial() -> None:
    pred = _segs([(0, 0.5, "a")])
    gt = _segs([(0, 1, "a")])
    assert math.isclose(temporal_iou(pred, gt)["mean_gt_iou"], 0.5)


def test_kendall_tau_reversed() -> None:
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == -1.0


def test_damerau_levenshtein_transposition() -> None:
    assert damerau_levenshtein(["a", "b"], ["b", "a"]) == 1
    assert damerau_levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_ordering_with_swapped_segments() -> None:
    pred = _segs([(0, 1, None), (1, 2, None), (2, 3, None)])
    # labels assigned by overlap should recover GT order -> tau 1.0
    result = ordering_metrics(pred, GT)
    assert result["kendall_tau"] == 1.0
