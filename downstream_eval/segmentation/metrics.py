# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Segmentation-quality metrics.

Predicted and ground-truth segmentations are lists of :class:`Segment` (seconds).
All metrics are pure-numpy / pure-python so they can be unit tested without GPUs.

Groups
------
- Boundary detection: ``boundary_f1``, ``boundary_mae`` (within a tolerance window).
- Segment overlap:    ``temporal_iou``.
- Completeness:       ``over_segmentation_rate``, ``under_segmentation_rate``, ``fragmentation_index``.
- Temporal ordering:  ``kendall_tau``, ``damerau_levenshtein`` on label sequences.
"""

import math

from downstream_eval.common.types import Segment


def segment_boundaries(segments: list[Segment]) -> list[float]:
    """Return sorted internal boundary timestamps for a segmentation.

    The global start (min start) and global end (max end) are excluded because
    they are shared trivially by any segmentation covering the same span.
    """
    if not segments:
        return []
    points = sorted({s.start for s in segments} | {s.end for s in segments})
    lo, hi = points[0], points[-1]
    return [p for p in points if lo < p < hi]


def _match_boundaries(
    pred: list[float], gt: list[float], tolerance: float
) -> tuple[int, int, int, list[tuple[int, int, float]]]:
    """Greedily match predicted to GT boundaries within ``tolerance`` (nearest first)."""
    pairs = sorted(
        (abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p - g) <= tolerance
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for dist, i, j in pairs:
        if i not in used_p and j not in used_g:
            matched.append((i, j, dist))
            used_p.add(i)
            used_g.add(j)
    tp = len(matched)
    return tp, len(pred) - tp, len(gt) - tp, matched


def boundary_f1(pred_segments: list[Segment], gt_segments: list[Segment], tolerance: float = 0.5) -> dict[str, float]:
    """Boundary precision / recall / F1 with a temporal tolerance window.

    Args:
        pred_segments: Predicted segmentation.
        gt_segments: Ground-truth segmentation.
        tolerance: Max distance (seconds) for a predicted boundary to count as a hit.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``, ``tolerance_s``.

    """
    pred = segment_boundaries(pred_segments)
    gt = segment_boundaries(gt_segments)
    tp, fp, fn, _ = _match_boundaries(pred, gt, tolerance)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tolerance_s": tolerance,
    }


def boundary_mae(pred_segments: list[Segment], gt_segments: list[Segment]) -> float:
    """Mean absolute error (seconds) from each GT boundary to its nearest predicted boundary.

    Returns ``nan`` when either side has no internal boundaries.
    """
    pred = segment_boundaries(pred_segments)
    gt = segment_boundaries(gt_segments)
    if not pred or not gt:
        return float("nan")
    return sum(min(abs(g - p) for p in pred) for g in gt) / len(gt)


def _iou(a: Segment, b: Segment) -> float:
    """Temporal IoU between two segments."""
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def temporal_iou(pred_segments: list[Segment], gt_segments: list[Segment]) -> dict[str, float]:
    """Mean temporal IoU between predicted and GT segments.

    Returns:
        ``mean_gt_iou`` (avg over GT of best IoU with any prediction) and
        ``mean_matched_iou`` (avg IoU over a one-to-one greedy matching).

    """
    if not pred_segments or not gt_segments:
        return {"mean_gt_iou": 0.0, "mean_matched_iou": 0.0}
    iou = [[_iou(p, g) for g in gt_segments] for p in pred_segments]
    mean_gt_iou = sum(max(iou[i][j] for i in range(len(pred_segments))) for j in range(len(gt_segments)))
    mean_gt_iou /= len(gt_segments)

    triples = sorted(
        ((iou[i][j], i, j) for i in range(len(pred_segments)) for j in range(len(gt_segments))),
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched_vals: list[float] = []
    for val, i, j in triples:
        if i not in used_p and j not in used_g:
            matched_vals.append(val)
            used_p.add(i)
            used_g.add(j)
    mean_matched_iou = sum(matched_vals) / len(matched_vals) if matched_vals else 0.0
    return {"mean_gt_iou": mean_gt_iou, "mean_matched_iou": mean_matched_iou}


def completeness(
    pred_segments: list[Segment], gt_segments: list[Segment], tolerance: float = 0.5
) -> dict[str, float]:
    """Over/under-segmentation rates and fragmentation index.

    - ``over_segmentation_rate``  = spurious predicted boundaries / predicted boundaries (1 - precision).
    - ``under_segmentation_rate`` = missed GT boundaries / GT boundaries (1 - recall).
    - ``segment_count_ratio``     = #predicted segments / #GT segments (1.0 ideal).
    - ``fragmentation_index``     = mean #predicted segments overlapping each GT segment (1.0 ideal).
    """
    pred = segment_boundaries(pred_segments)
    gt = segment_boundaries(gt_segments)
    tp, fp, fn, _ = _match_boundaries(pred, gt, tolerance)
    over = fp / (tp + fp) if (tp + fp) else 0.0
    under = fn / (tp + fn) if (tp + fn) else 0.0
    count_ratio = len(pred_segments) / len(gt_segments) if gt_segments else float("nan")

    if gt_segments:
        overlaps = [sum(1 for p in pred_segments if _iou(p, g) > 0.0) for g in gt_segments]
        fragmentation = sum(overlaps) / len(overlaps)
    else:
        fragmentation = float("nan")

    return {
        "over_segmentation_rate": over,
        "under_segmentation_rate": under,
        "segment_count_ratio": count_ratio,
        "fragmentation_index": fragmentation,
    }


def label_segments_by_overlap(pred_segments: list[Segment], gt_segments: list[Segment]) -> list[str | None]:
    """Assign each predicted segment the label of the GT segment it most overlaps."""
    labels: list[str | None] = []
    for p in pred_segments:
        best_overlap = 0.0
        best_label: str | None = None
        for g in gt_segments:
            overlap = max(0.0, min(p.end, g.end) - max(p.start, g.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = g.label
        labels.append(best_label)
    return labels


def kendall_tau(seq_a: list[object], seq_b: list[object]) -> float:
    """Kendall's tau-b rank correlation between the orderings of two label sequences.

    Only labels present in *both* sequences are compared, using each label's first
    occurrence index as its rank. Returns ``nan`` if fewer than two common labels.
    """
    pos_a = {x: i for i, x in enumerate(seq_a) if x is not None}
    pos_b = {x: i for i, x in enumerate(seq_b) if x is not None}
    common = [x for x in pos_a if x in pos_b]
    n = len(common)
    if n < 2:
        return float("nan")
    xs = [pos_a[x] for x in common]
    ys = [pos_b[x] for x in common]
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom > 0 else float("nan")


def damerau_levenshtein(seq_a: list[object], seq_b: list[object]) -> int:
    """Optimal-string-alignment Damerau-Levenshtein distance (adjacent transpositions)."""
    len_a, len_b = len(seq_a), len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a
    dist = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dist[i][0] = i
    for j in range(len_b + 1):
        dist[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and seq_a[i - 1] == seq_b[j - 2] and seq_a[i - 2] == seq_b[j - 1]:
                dist[i][j] = min(dist[i][j], dist[i - 2][j - 2] + 1)
    return dist[len_a][len_b]


def ordering_metrics(pred_segments: list[Segment], gt_segments: list[Segment]) -> dict[str, float]:
    """Temporal-ordering metrics from labelled segments.

    Predicted segments are labelled by maximum overlap with GT segments, then the
    predicted label order is compared against the GT label order.
    """
    pred_labels = [lbl for lbl in label_segments_by_overlap(pred_segments, gt_segments) if lbl is not None]
    gt_labels = [s.label for s in gt_segments if s.label is not None]
    tau = kendall_tau(pred_labels, gt_labels)  # type: ignore[arg-type]
    dl = damerau_levenshtein(pred_labels, gt_labels)  # type: ignore[arg-type]
    norm = max(len(pred_labels), len(gt_labels)) or 1
    return {
        "kendall_tau": tau,
        "damerau_levenshtein": float(dl),
        "damerau_levenshtein_normalized": dl / norm,
    }


def evaluate_segmentation(
    pred_segments: list[Segment], gt_segments: list[Segment], tolerance: float = 0.5
) -> dict[str, float]:
    """Compute the full segmentation-quality metric suite for one video."""
    out: dict[str, float] = {}
    out.update({f"boundary_{k}": v for k, v in boundary_f1(pred_segments, gt_segments, tolerance).items()})
    out["boundary_mae_s"] = boundary_mae(pred_segments, gt_segments)
    out.update(temporal_iou(pred_segments, gt_segments))
    out.update(completeness(pred_segments, gt_segments, tolerance))
    out.update(ordering_metrics(pred_segments, gt_segments))
    return out
