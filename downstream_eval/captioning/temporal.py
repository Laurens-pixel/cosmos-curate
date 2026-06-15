# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporal-coherence caption metrics.

- ``narrative_coherence``        : logical flow across consecutive captions. Uses an LLM judge
                                   callable when supplied; otherwise a lexical-continuity fallback.
- ``temporal_grounding_accuracy``: does each window's caption describe the right moment?
- ``action_order_consistency``   : Kendall's tau between predicted and GT action order in captions.
"""

from collections.abc import Callable

from downstream_eval.captioning.reference_metrics import tokenize
from downstream_eval.segmentation.metrics import kendall_tau

_STOPWORDS = frozenset(
    "a an the is are was were be been being of to in on at and or with for from this that it its "
    "as by into over then while during after before person hand video frame scene shows showing".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS}


def lexical_coherence(captions: list[str]) -> float:
    """Mean Jaccard overlap of content tokens between consecutive captions.

    A cheap, model-free proxy for narrative continuity: adjacent captions in a coherent
    narrative share some entities/objects but also introduce new content.
    """
    if len(captions) < 2:
        return float("nan")
    overlaps: list[float] = []
    for prev, curr in zip(captions, captions[1:], strict=False):
        a, b = _content_tokens(prev), _content_tokens(curr)
        union = a | b
        overlaps.append(len(a & b) / len(union) if union else 0.0)
    return sum(overlaps) / len(overlaps)


def narrative_coherence(
    captions: list[str],
    judge: Callable[[list[str]], float] | None = None,
) -> dict[str, float | str]:
    """Narrative coherence across a caption sequence.

    Args:
        captions: Consecutive captions for one video, in temporal order.
        judge: Optional callable returning a 0-1 coherence score for the sequence
            (e.g. an LLM-as-judge). When ``None``, a lexical-continuity fallback is used.

    Returns:
        Dict with ``coherence`` and the ``backend`` used.

    """
    if judge is not None:
        return {"coherence": float(judge(captions)), "backend": "llm-judge"}
    return {"coherence": lexical_coherence(captions), "backend": "lexical-fallback"}


def _match_label(caption: str, label_vocabulary: list[str]) -> str | None:
    """Predict the closest label for a caption by content-token overlap."""
    caption_tokens = _content_tokens(caption)
    if not caption_tokens:
        return None
    best_label: str | None = None
    best_score = 0
    for label in label_vocabulary:
        score = len(caption_tokens & _content_tokens(label))
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def temporal_grounding_accuracy(
    windows: list[tuple[float, float, str]],
    gt_intervals: list[tuple[float, float, str]],
) -> dict[str, float]:
    """Fraction of windows whose caption matches the GT action overlapping that window.

    Args:
        windows: ``(start_s, end_s, caption)`` per captioning window, in time order.
        gt_intervals: ``(start_s, end_s, label)`` ground-truth action intervals.

    Returns:
        Dict with ``temporal_grounding_accuracy`` and ``num_windows``.

    """
    if not windows or not gt_intervals:
        return {"temporal_grounding_accuracy": float("nan"), "num_windows": float(len(windows))}
    vocabulary = sorted({lbl for _, _, lbl in gt_intervals})
    correct = 0
    for w_start, w_end, caption in windows:
        best_overlap = 0.0
        gt_label: str | None = None
        for g_start, g_end, label in gt_intervals:
            overlap = max(0.0, min(w_end, g_end) - max(w_start, g_start))
            if overlap > best_overlap:
                best_overlap = overlap
                gt_label = label
        if gt_label is not None and _match_label(caption, vocabulary) == gt_label:
            correct += 1
    return {"temporal_grounding_accuracy": correct / len(windows), "num_windows": float(len(windows))}


def action_order_consistency(
    predicted_labels: list[str], gt_labels: list[str]
) -> dict[str, float]:
    """Kendall's tau between the predicted and GT action label order in captions."""
    return {"action_order_kendall_tau": kendall_tau(list(predicted_labels), list(gt_labels))}
