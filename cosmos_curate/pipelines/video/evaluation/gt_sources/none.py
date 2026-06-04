# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Null GT source — always returns empty GT.

Use this when the judge is reference-free (e.g. VLM judges that score quality from
video alone) or when GT is being added through a different pathway.
"""

from typing import Any

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource


class NoneGt(GtSource):
    """A GT source that always returns no GT."""

    @staticmethod
    def name() -> str:
        """Return the GT source identifier."""
        return "none"

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Always return empty GT."""
        return "", {}
