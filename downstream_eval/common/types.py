# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight data structures shared across evaluators.

These use stdlib :mod:`dataclasses` (not ``attrs``) so the suite has no third-party
dependency beyond numpy and can be imported and tested in isolation.
"""

from dataclasses import dataclass, field


@dataclass
class Segment:
    """A temporal segment in seconds, optionally carrying a (sub-task) label.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        label: Optional sub-task / action label used by ordering and recognition metrics.

    """

    start: float
    end: float
    label: str | None = None

    @property
    def duration(self) -> float:
        """Return the segment duration in seconds (clamped at 0)."""
        return max(0.0, self.end - self.start)


@dataclass
class WindowRecord:
    """One captioning window inside a clip."""

    start_frame: int
    end_frame: int
    captions: dict[str, str] = field(default_factory=dict)
    judge: dict[str, dict[str, object]] = field(default_factory=dict)

    def primary_caption(self) -> str:
        """Return the first available caption text, or empty string."""
        for value in self.captions.values():
            if value:
                return value
        return ""


@dataclass
class ClipRecord:
    """A curated clip with its source video, time span, windows, and optional embedding."""

    uuid: str
    source_video: str
    span: tuple[float, float]
    windows: list[WindowRecord] = field(default_factory=list)
    framerate: float | None = None

    def to_segment(self) -> Segment:
        """Return the clip's time span as a :class:`Segment`."""
        return Segment(start=self.span[0], end=self.span[1])

    def all_captions(self) -> list[str]:
        """Return every non-empty caption across all windows."""
        out: list[str] = []
        for window in self.windows:
            out.extend(text for text in window.captions.values() if text)
        return out
