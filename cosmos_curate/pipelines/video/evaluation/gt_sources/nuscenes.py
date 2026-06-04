# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nuScenes GT source.

Reads the ``scene.json`` file from a nuScenes v1.0 metadata directory and
maps scene names to their human-written scene description tags::

    scene-0061_CAM_FRONT.mp4  ->  "Parked truck, construction, intersection, ..."
    scene-0061_CAM_FRONT_LEFT.mp4  ->  same description (scene-level GT)
    scene-0061_CAM_FRONT_RIGHT.mp4  ->  same description (scene-level GT)

The description is the same for all cameras of a scene because nuScenes
annotates at scene granularity, not camera granularity.  In multi-view mode
only the CAM_FRONT window carries a caption, so judgement runs once per scene.

The ``start_frame`` / ``end_frame`` arguments are unused: the description
covers the entire ~20 s scene clip.
"""

import json
import pathlib
import re
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource

# Matches filenames like scene-0061_CAM_FRONT.mp4 or scene-0061_CAM_FRONT_LEFT.mp4
_SCENE_RE = re.compile(r"^(scene-\d+)_CAM_")


class NuScenesGt(GtSource):
    """Look up the nuScenes scene-level description for a camera clip."""

    def __init__(self, scene_json_path: pathlib.Path) -> None:
        """Prepare the GT source; loading is deferred to first lookup."""
        self._scene_json_path = pathlib.Path(scene_json_path)
        self._descriptions: dict[str, str] = {}
        self._loaded = False

    @staticmethod
    def name() -> str:
        """Return the GT source identifier."""
        return "nuscenes"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._scene_json_path.exists():
            logger.warning(f"NuScenesGt: scene.json not found at {self._scene_json_path}")
            return
        try:
            scenes: list[dict[str, Any]] = json.loads(self._scene_json_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"NuScenesGt: failed to load {self._scene_json_path}: {exc}")
            return
        for entry in scenes:
            name = entry.get("name", "")
            description = entry.get("description", "")
            if name and description:
                self._descriptions[name] = description
        logger.info(f"NuScenesGt: loaded {len(self._descriptions)} scene descriptions")

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return the scene description for the clip; frame args unused."""
        self._ensure_loaded()
        stem = pathlib.Path(video_name).stem  # e.g. "scene-0061_CAM_FRONT"
        m = _SCENE_RE.match(stem)
        if not m:
            return "", {}
        scene_name = m.group(1)  # e.g. "scene-0061"
        description = self._descriptions.get(scene_name, "")
        if not description:
            return "", {}
        return description, {"scene_name": scene_name}
