# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""YouCook2 GT caption source.

Reads the ``gt_captions.json`` file produced by ``download_youcook2.py``.
Each entry is keyed by ``<youtube_id>_<segment_index>`` and contains a
``"sentence"`` field with the human-annotated description::

    {
      "xHr8X2Wpmno_0": {
        "youtube_id": "xHr8X2Wpmno",
        "sentence": "pick the ends off the verdalago",
        "segment_sec": [47.0, 60.0],
        ...
      },
      ...
    }

Clip filenames are expected to be ``<youtube_id>_<segment_index>.mp4``.
The ``start_frame`` / ``end_frame`` arguments are unused because each clip
is already a single annotated segment (whole clip = one window).
"""

import json
import pathlib
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource


class YouCook2Gt(GtSource):
    """Look up the human-annotated GT sentence for a YouCook2 clip."""

    def __init__(self, gt_captions_path: pathlib.Path) -> None:
        """Load and index the gt_captions.json file."""
        self._gt_captions_path = pathlib.Path(gt_captions_path)
        self._captions: dict[str, str] = {}
        self._splits: dict[str, str] = {}
        self._loaded = False

    @staticmethod
    def name() -> str:
        """Return the GT source identifier."""
        return "youcook2"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._gt_captions_path.exists():
            logger.warning(f"YouCook2Gt: gt_captions.json not found at {self._gt_captions_path}")
            return
        try:
            data: dict[str, Any] = json.loads(self._gt_captions_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"YouCook2Gt: failed to load {self._gt_captions_path}: {exc}")
            return
        for key, entry in data.items():
            sentence = entry.get("sentence", "")
            if sentence:
                self._captions[key] = str(sentence)
                self._splits[key] = str(entry.get("split", ""))
        logger.info(f"YouCook2Gt: loaded {len(self._captions)} GT captions from {self._gt_captions_path}")

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return the GT sentence for the clip; frame args unused (one segment per clip)."""
        self._ensure_loaded()
        stem = pathlib.Path(video_name).stem  # e.g. "xHr8X2Wpmno_0"
        sentence = self._captions.get(stem, "")
        if not sentence:
            return "", {}
        return sentence, {"split": self._splits.get(stem, "")}
