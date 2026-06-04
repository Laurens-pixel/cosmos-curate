# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manual annotation JSON GT source.

Reads a single JSON file with a top-level ``windows`` list, each entry of the form::

    {
      "source_video": "327_685046_head_color.mp4",
      "window_start_s": 0.5,
      "window_end_s": 9.0,
      "gt_action_text": "Retrieve shiitake mushroom from the shelf.",
      "correct": "incorrect"        # optional, mirrors annotation_*_ground_truth.json
    }

Lookup is by ``(source_video, round(start_s, 3))``. The native frame rate is required
to convert the pipeline's ``start_frame``/``end_frame`` to seconds.
"""

import json
import pathlib
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource


class ManualAnnotationJsonGt(GtSource):
    """Load GT from a manually-annotated JSON file (e.g. annotation_ground_truth.json)."""

    def __init__(self, annotation_path: pathlib.Path, native_fps: float = 30.0) -> None:
        """Read all entries into an in-memory map keyed by (video, rounded_start_s)."""
        self._native_fps = native_fps
        path = pathlib.Path(annotation_path)
        if not path.exists():
            msg = f"ManualAnnotationJsonGt: annotation file not found: {path}"
            raise FileNotFoundError(msg)

        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"ManualAnnotationJsonGt: failed to load {path}: {exc}"
            raise RuntimeError(msg) from exc

        windows = data.get("windows", [])
        self._lookup: dict[tuple[str, float], dict[str, Any]] = {}
        for w in windows:
            video = w.get("source_video", "")
            start_s = w.get("window_start_s")
            if not video or start_s is None:
                continue
            self._lookup[(video, round(float(start_s), 3))] = w
        logger.info(f"ManualAnnotationJsonGt: loaded {len(self._lookup)} GT entries from {path}")

    @staticmethod
    def name() -> str:
        """Return the GT source identifier."""
        return "manual"

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return GT action text matched by start-time, or empty if not annotated."""
        start_s = round(start_frame / self._native_fps, 3)
        entry = self._lookup.get((video_name, start_s))
        if entry is None:
            return "", {}
        gt_text = str(entry.get("gt_action_text", ""))
        extras = {k: v for k, v in entry.items() if k != "gt_action_text"}
        return gt_text, extras
