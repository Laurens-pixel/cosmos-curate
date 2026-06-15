# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action-recognition / classification metrics.

Works from either a class-score matrix ``(N, C)`` (enables Top-K) or hard predicted
labels (Top-1 only). Metrics: Top-1 / Top-5 accuracy, mean per-class accuracy
(handles class imbalance), and macro-F1. Pure-numpy.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


def topk_accuracy(scores: FloatArray, labels: IntArray, k: int) -> float:
    """Top-K accuracy from a class-score matrix."""
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) == 0:
        return float("nan")
    k = min(k, scores.shape[1])
    topk = np.argsort(-scores, axis=1)[:, :k]
    hits = np.any(topk == labels[:, None], axis=1)
    return float(hits.mean())


def mean_per_class_accuracy(y_true: IntArray, y_pred: IntArray, num_classes: int) -> float:
    """Average of per-class recall (robust to class imbalance)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    accuracies: list[float] = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            accuracies.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(accuracies)) if accuracies else float("nan")


def macro_f1(y_true: IntArray, y_pred: IntArray, num_classes: int) -> float:
    """Unweighted mean F1 across classes."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def evaluate_action_recognition(
    labels: IntArray,
    scores: FloatArray | None = None,
    predictions: IntArray | None = None,
    num_classes: int | None = None,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute the action-recognition metric suite.

    Provide ``scores`` (for Top-K) and/or ``predictions`` (hard labels). When only
    ``scores`` is given, ``predictions`` is taken as its argmax.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if scores is None and predictions is None:
        msg = "Provide either 'scores' or 'predictions'."
        raise ValueError(msg)
    if predictions is None:
        predictions = np.argmax(np.asarray(scores), axis=1).astype(np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if num_classes is None:
        num_classes = int(max(labels.max(), predictions.max())) + 1

    out: dict[str, float] = {
        "top1_accuracy": float((predictions == labels).mean()) if len(labels) else float("nan"),
        "mean_per_class_accuracy": mean_per_class_accuracy(labels, predictions, num_classes),
        "macro_f1": macro_f1(labels, predictions, num_classes),
        "num_samples": float(len(labels)),
    }
    if scores is not None:
        for k in ks:
            out[f"top{k}_accuracy"] = topk_accuracy(scores, labels, k)
    return out
