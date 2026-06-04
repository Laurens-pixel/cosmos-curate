# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GT source ABC.

Each GT source knows how to map ``(video_name, frame_range)`` to a ground-truth
action description string. It is orthogonal to the judge model: a Gemma4 text judge
can be paired with any GT source, including ``NoneGt`` for reference-free use.
"""

from abc import ABC, abstractmethod
from typing import Any


class GtSource(ABC):
    """Look up ground-truth action text for a window."""

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Return the unique GT source identifier (e.g. ``agibot``, ``manual``, ``none``)."""

    @abstractmethod
    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return ``(gt_action_text, extras)`` for the given window.

        ``gt_action_text`` is ``""`` if no GT is available — judge plugins should treat
        empty GT as reference-free or skip the item.

        ``extras`` is a free-form dict that is attached to the resulting
        ``Window.judge[variant]["gt_extras"]`` so e.g. ``gt_skill`` can flow through
        without coupling the ABC to any one dataset's schema.
        """
