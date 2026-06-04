# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgiBotWorld task_info GT source.

Reads ``task_info/task_<id>.json`` files. Each file is a list of episode entries
with ``label_info.action_config`` listing per-frame-range actions::

    [
      {
        "episode_id": 685046,
        "label_info": {
          "action_config": [
            {"start_frame": 8, "end_frame": 218,
             "action_text": "Retrieve cucumber from the shelf.",
             "skill": "Pick"},
            ...
          ]
        }
      },
      ...
    ]

Video filenames are expected to be ``<task_id>_<episode_id>_head_color.mp4``.
"""

import json
import pathlib
import re
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource

_VIDEO_NAME_RE = re.compile(r"^(\d+)_(\d+)_head_color\.mp4$")


class AgibotTaskInfoGt(GtSource):
    """Look up GT action text from per-task action_config JSONs."""

    def __init__(self, task_info_dir: pathlib.Path) -> None:
        """Cache action_config lists keyed by ``(task_id, episode_id)``."""
        self._task_info_dir = pathlib.Path(task_info_dir)
        # episode_actions[(task_id, episode_id)] = list[action_config_entry]
        self._episode_actions: dict[tuple[str, str], list[dict[str, Any]]] = {}
        # Tasks we've already tried to load (success or failure)
        self._loaded_tasks: set[str] = set()

    @staticmethod
    def name() -> str:
        """Return the GT source identifier."""
        return "agibot"

    def _load_task(self, task_id: str) -> None:
        if task_id in self._loaded_tasks:
            return
        self._loaded_tasks.add(task_id)
        path = self._task_info_dir / f"task_{task_id}.json"
        if not path.exists():
            logger.warning(f"AgibotTaskInfoGt: task_info file missing for task {task_id}: {path}")
            return
        try:
            episodes = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"AgibotTaskInfoGt: failed to load {path}: {exc}")
            return
        for ep in episodes:
            ep_id = str(ep.get("episode_id", ""))
            if not ep_id:
                continue
            actions = ep.get("label_info", {}).get("action_config", [])
            self._episode_actions[(task_id, ep_id)] = actions

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return action_text whose annotated frame range overlaps the window most."""
        m = _VIDEO_NAME_RE.match(pathlib.Path(video_name).name)
        if not m:
            return "", {}
        task_id, episode_id = m.group(1), m.group(2)

        self._load_task(task_id)
        actions = self._episode_actions.get((task_id, episode_id))
        if not actions:
            return "", {}

        best_text = ""
        best_skill = ""
        best_overlap = 0
        for entry in actions:
            s = int(entry.get("start_frame", 0))
            e = int(entry.get("end_frame", 0))
            overlap = max(0, min(e, end_frame) - max(s, start_frame))
            if overlap > best_overlap:
                best_overlap = overlap
                best_text = str(entry.get("action_text", ""))
                best_skill = str(entry.get("skill", ""))

        extras = {
            "gt_skill": best_skill,
            "task_id": task_id,
            "episode_id": episode_id,
        } if best_text else {}
        return best_text, extras
