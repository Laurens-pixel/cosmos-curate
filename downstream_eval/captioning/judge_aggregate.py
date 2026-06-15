# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the in-pipeline VLM-as-judge outputs into caption-quality summaries.

The judge runs inside the pipeline (``JudgeStage``) and writes, per window per
``judge_variant``::

    {"verdict": "CORRECT"|"INCORRECT"|..., "score": int, "explanation": str,
     "gt_action_text": str, "gt_extras": {...}, ...}

This module summarises those verdicts (no model is re-run here). When ``gt_extras``
carries a human correctness label, judge-vs-human agreement (precision/recall/F1) is
also reported.
"""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from downstream_eval.common.types import ClipRecord

_CORRECT = "CORRECT"
_HUMAN_LABEL_KEYS = ("human_correct", "is_correct", "correct")


def iter_judge_records(judgments: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield ``(variant, record)`` pairs from a loaded ``all_window_judgments.json``."""
    for clips in judgments.values():
        for windows in clips.values():
            for variants in windows.values():
                if isinstance(variants, dict):
                    for variant, record in variants.items():
                        if isinstance(record, dict):
                            yield {"variant": variant, **record}


def iter_judge_records_from_clips(clips: list[ClipRecord]) -> Iterable[dict[str, Any]]:
    """Yield ``(variant, record)`` pairs from per-clip window judge dicts."""
    for clip in clips:
        for window in clip.windows:
            for variant, record in window.judge.items():
                if isinstance(record, dict):
                    yield {"variant": variant, **record}


def _human_label(record: dict[str, Any]) -> bool | None:
    extras = record.get("gt_extras") or {}
    for key in _HUMAN_LABEL_KEYS:
        if key in extras:
            return bool(extras[key])
    return None


def aggregate_judgments(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarise judge verdicts per ``judge_variant``.

    Returns:
        ``{variant: {num, correct_rate, mean_score, verdict_distribution, [precision/recall/f1]}}``.
        Agreement metrics are included only when human correctness labels are present.

    """
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_variant.setdefault(record.get("variant", "default"), []).append(record)

    out: dict[str, dict[str, Any]] = {}
    for variant, recs in by_variant.items():
        verdicts = [r.get("verdict") for r in recs if r.get("verdict") is not None]
        scores = [float(r["score"]) for r in recs if isinstance(r.get("score"), (int, float))]
        distribution = dict(Counter(verdicts))
        summary: dict[str, Any] = {
            "num": len(recs),
            "num_judged": len(verdicts),
            "correct_rate": (sum(1 for v in verdicts if v == _CORRECT) / len(verdicts)) if verdicts else float("nan"),
            "mean_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "verdict_distribution": distribution,
        }

        pairs = [(_human_label(r), r.get("verdict") == _CORRECT) for r in recs if _human_label(r) is not None]
        if pairs:
            tp = sum(1 for human, pred in pairs if human and pred)
            fp = sum(1 for human, pred in pairs if not human and pred)
            fn = sum(1 for human, pred in pairs if human and not pred)
            tn = sum(1 for human, pred in pairs if not human and not pred)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            summary.update(
                {
                    "human_agreement": (tp + tn) / len(pairs),
                    "precision_vs_human": precision,
                    "recall_vs_human": recall,
                    "f1_vs_human": f1,
                    "num_human_labeled": len(pairs),
                }
            )
        out[variant] = summary
    return out
