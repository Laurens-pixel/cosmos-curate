# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""InHARD GT source.

The InHARD dataset's ``Segmented/RGBSegmented/`` directory organises pre-cut action clips
into one subdirectory per action class::

    Segmented/RGBSegmented/Take screwdriver/P01_R01_0012.56_0016.88.mp4
    Segmented/RGBSegmented/Assemble system/P02_R03_0041.20_0044.56.mp4

The action class label is therefore the **parent directory name** of the video file —
no separate annotation file is needed.  ``start_frame`` / ``end_frame`` are unused
because each clip already spans exactly one action.
"""

import pathlib
from typing import Any

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource


class InHardGt(GtSource):
    """Look up the InHARD action class from the clip's parent directory name."""

    @staticmethod
    def name() -> str:
        """Return the GT source identifier."""
        return "inhard"

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return the action class label (parent dir name); frame args unused."""
        action_class = pathlib.Path(video_name).parent.name
        if not action_class:
            return "", {}
        return action_class, {"action_class": action_class}
