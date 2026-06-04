# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""InHARD Online GT source.

Reads Anvil XML annotation files from the InHARD ``Online/Labels/`` directory.
Each ``.anvil`` file is UTF-16 encoded XML with time-stamped action labels::

    <el index="0" start="2.36" end="3.28">
        <attribute name="type">[OP010] Consult sheets</attribute>
    </el>

The annotation with the greatest temporal overlap with the query window is returned.
Gaps between annotations (transitions / "No action" periods) return an empty string.
"""

import pathlib
import xml.etree.ElementTree as ET
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource

_INHARD_FPS = 30000 / 1001  # ~29.97 — true frame rate of InHARD Online RGB videos


class InHardOnlineGt(GtSource):
    """Look up the InHARD action label from Anvil annotation files.

    Parameters
    ----------
    labels_dir:
        Path to the directory containing ``.anvil`` files (one per video).
    fps:
        Frame rate used to convert frame indices to seconds.  Defaults to the
        native InHARD frame rate (30000/1001 ≈ 29.97 fps).
    """

    def __init__(self, labels_dir: pathlib.Path, fps: float = _INHARD_FPS) -> None:
        self._labels_dir = pathlib.Path(labels_dir)
        self._fps = fps
        # Cache: video_stem → list of (start_s, end_s, label)
        self._cache: dict[str, list[tuple[float, float, str]]] = {}

    @staticmethod
    def name() -> str:
        """Return the GT source identifier."""
        return "inhard_online"

    def _load(self, stem: str) -> list[tuple[float, float, str]]:
        if stem in self._cache:
            return self._cache[stem]
        path = self._labels_dir / f"{stem}.anvil"
        if not path.exists():
            logger.warning(f"InHardOnlineGt: annotation file not found: {path}")
            self._cache[stem] = []
            return []
        try:
            with open(path, encoding="utf-16") as fh:
                content = fh.read()
            root = ET.fromstring(content)
            track = root.find('.//track[@name="Action Label"]')
            entries: list[tuple[float, float, str]] = []
            if track is not None:
                for el in track.findall("el"):
                    attr = el.find('attribute[@name="type"]')
                    if attr is None or not attr.text:
                        continue
                    entries.append((float(el.get("start", 0)), float(el.get("end", 0)), attr.text))
            self._cache[stem] = entries
        except Exception as exc:
            logger.error(f"InHardOnlineGt: failed to parse {path}: {exc}")
            self._cache[stem] = []
        return self._cache[stem]

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return the action label with the most overlap with the query window."""
        stem = pathlib.Path(video_name).stem
        entries = self._load(stem)
        if not entries:
            return "", {}

        win_start_s = start_frame / self._fps
        win_end_s = end_frame / self._fps

        best_label = ""
        best_overlap = 0.0
        for ann_start, ann_end, label in entries:
            overlap = max(0.0, min(ann_end, win_end_s) - max(ann_start, win_start_s))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label

        if not best_label:
            return "", {}
        return best_label, {"annotation_overlap_s": round(best_overlap, 3)}
