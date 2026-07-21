# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wearable AI (facebook/wearable-ai, EgoConv) GT source.

The EgoConv dataset contains egocentric conversation clips.  There are **no
per-clip action annotations** — only ``clip_index``, ``video_filename``,
``source_repo``, and ``source_path`` in the manifest.

``lookup()`` always returns ``("", {})`` so caption metrics that require GT
labels (action_f1, semantic grounding) are simply skipped for this dataset
while reference-free metrics (judge faithfulness, object grounding,
consistency) still apply.
"""

from typing import Any

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource


class WearableAiGt(GtSource):
    """No-op GT source for the EgoConv / Wearable AI dataset (no action labels)."""

    @staticmethod
    def name() -> str:
        return "wearable_ai"

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        return "", {}
