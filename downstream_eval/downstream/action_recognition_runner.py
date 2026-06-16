# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action-recognition *task runner*: classifies clips, then evaluates.

Unlike :mod:`downstream_eval.downstream.action_recognition` (metrics only), this module
actually *performs* recognition on the curated clips with three training-free / lightweight
classifiers, then scores predictions with the metric suite:

- ``run_nearest_centroid``  : nearest class-centroid in clip-embedding space (train/test split).
- ``run_linear_probe``      : logistic-regression linear probe on clip embeddings (sklearn).
- ``run_zero_shot_language``: zero-shot via language — assign each clip the class whose name is
  most similar to the clip's generated caption (no labels at train time).

All three return predictions/scores plus the metric dict, so they slot directly into a paper
table.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from downstream_eval.downstream.action_recognition import evaluate_action_recognition
from downstream_eval.downstream.encoders import TextEncoder, build_text_encoder

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass
class RecognitionRun:
    """Result of actually running an action-recognition task."""

    method: str
    metrics: dict[str, float]
    predictions: IntArray
    class_names: list[str]
    test_indices: list[int] = field(default_factory=list)


def _stratified_split(labels: IntArray, train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Per-class stratified train/test split of sample indices."""
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_train = max(1, int(round(len(idx) * train_frac))) if len(idx) > 1 else 1
        train.extend(idx[:n_train].tolist())
        test.extend(idx[n_train:].tolist() if len(idx) > 1 else idx.tolist())
    return sorted(train), sorted(test)


def run_nearest_centroid(
    embeddings: FloatArray,
    labels: IntArray,
    class_names: list[str],
    train_frac: float = 0.6,
    seed: int = 0,
    ks: tuple[int, ...] = (1, 5),
) -> RecognitionRun:
    """Classify clips by nearest class-centroid in embedding space."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    train_idx, test_idx = _stratified_split(labels, train_frac, seed)

    def _norm(mat: FloatArray) -> FloatArray:
        return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)

    num_classes = len(class_names)
    centroids = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float32)
    for cls in range(num_classes):
        members = [i for i in train_idx if labels[i] == cls]
        if members:
            centroids[cls] = embeddings[members].mean(axis=0)

    scores = _norm(embeddings[test_idx]) @ _norm(centroids).T
    metrics = evaluate_action_recognition(
        labels[test_idx], scores=scores.astype(np.float32), num_classes=num_classes, ks=ks
    )
    return RecognitionRun(
        method="nearest_centroid",
        metrics=metrics,
        predictions=np.argmax(scores, axis=1).astype(np.int64),
        class_names=class_names,
        test_indices=test_idx,
    )


def run_linear_probe(
    embeddings: FloatArray,
    labels: IntArray,
    class_names: list[str],
    train_frac: float = 0.6,
    seed: int = 0,
    ks: tuple[int, ...] = (1, 5),
) -> RecognitionRun:
    """Train a logistic-regression linear probe on clip embeddings (requires scikit-learn)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        msg = "run_linear_probe requires scikit-learn. Use run_nearest_centroid instead."
        raise RuntimeError(msg) from exc

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    train_idx, test_idx = _stratified_split(labels, train_frac, seed)
    num_classes = len(class_names)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(embeddings[train_idx], labels[train_idx])

    present = list(clf.classes_)
    decision = clf.predict_proba(embeddings[test_idx])
    scores = np.zeros((len(test_idx), num_classes), dtype=np.float32)
    for col, cls in enumerate(present):
        scores[:, int(cls)] = decision[:, col]

    metrics = evaluate_action_recognition(labels[test_idx], scores=scores, num_classes=num_classes, ks=ks)
    return RecognitionRun(
        method="linear_probe",
        metrics=metrics,
        predictions=np.argmax(scores, axis=1).astype(np.int64),
        class_names=class_names,
        test_indices=test_idx,
    )


def run_zero_shot_language(
    captions: list[str],
    labels: IntArray,
    class_names: list[str],
    encoder: TextEncoder | None = None,
    ks: tuple[int, ...] = (1, 5),
) -> RecognitionRun:
    """Zero-shot recognition: pick the class name most similar to each clip's caption.

    No labels are used to fit anything; labels are only used to score. This measures whether
    generated captions carry enough class signal to recognise the action directly.
    """
    encoder = encoder or build_text_encoder("auto")
    labels = np.asarray(labels, dtype=np.int64)
    caption_vecs = encoder.encode(captions)
    class_vecs = encoder.encode(class_names)

    def _norm(mat: FloatArray) -> FloatArray:
        return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8, None)

    scores = (_norm(caption_vecs) @ _norm(class_vecs).T).astype(np.float32)
    metrics = evaluate_action_recognition(labels, scores=scores, num_classes=len(class_names), ks=ks)
    return RecognitionRun(
        method="zero_shot_language",
        metrics=metrics,
        predictions=np.argmax(scores, axis=1).astype(np.int64),
        class_names=class_names,
        test_indices=list(range(len(captions))),
    )
