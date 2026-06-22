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

# A window is a "transition" (spans ≥2 actions) when at least this many actions each
# cover this fraction of the window. Used to flag the multi-action case so downstream
# metrics can treat such windows differently from clean single-action windows.
_TRANSITION_MIN_WINDOW_COVERAGE = 0.15


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
        """Return GT for a window, with *all* overlapping actions exposed in ``extras``.

        A single 256-frame (~8.5 s) window routinely spans more than one annotated
        action (e.g. *pick* then *place*). The legacy behaviour assigned the single
        majority-overlap label, silently discarding minority actions — penalising
        captions that correctly describe a transition and rewarding captions that
        miss a boundary action.

        This implementation still returns the majority-overlap ``action_text`` as the
        primary string (backward compatible for text judges), but additionally records
        in ``extras``:

        * ``all_actions``   — every overlapping action with per-action coverage, sorted
          by overlap (most first). Each entry::

              {action_text, skill, start_frame, end_frame,
               overlap_frames, window_coverage, action_coverage}

          ``window_coverage`` = overlap / window_length (how much of the *window* this
          action explains); ``action_coverage`` = overlap / action_length (how much of
          the *action* falls inside the window).
        * ``window_coverage`` — fraction of the window covered by *any* GT action. A low
          value flags a window whose content is not described by the annotations (e.g. a
          novel/idle moment) so it can be excluded rather than forced onto a wrong label.
        * ``is_transition``  — True when ≥2 actions each cover ≥15% of the window.
        * ``num_overlapping_actions``.
        """
        m = _VIDEO_NAME_RE.match(pathlib.Path(video_name).name)
        if not m:
            return "", {}
        task_id, episode_id = m.group(1), m.group(2)

        self._load_task(task_id)
        actions = self._episode_actions.get((task_id, episode_id))
        if not actions:
            return "", {}

        window_len = max(1, end_frame - start_frame)
        overlapping: list[dict[str, Any]] = []
        for entry in actions:
            s = int(entry.get("start_frame", 0))
            e = int(entry.get("end_frame", 0))
            overlap = max(0, min(e, end_frame) - max(s, start_frame))
            if overlap <= 0:
                continue
            action_len = max(1, e - s)
            overlapping.append(
                {
                    "action_text": str(entry.get("action_text", "")),
                    "skill": str(entry.get("skill", "")),
                    "start_frame": s,
                    "end_frame": e,
                    "overlap_frames": overlap,
                    "window_coverage": round(overlap / window_len, 4),
                    "action_coverage": round(overlap / action_len, 4),
                }
            )

        if not overlapping:
            return "", {}

        overlapping.sort(key=lambda a: a["overlap_frames"], reverse=True)
        primary = overlapping[0]
        # Total window coverage counts each frame once (actions are non-overlapping in
        # AgiBotWorld annotations, so summing overlaps is exact here).
        total_coverage = min(1.0, sum(a["overlap_frames"] for a in overlapping) / window_len)
        n_significant = sum(1 for a in overlapping if a["window_coverage"] >= _TRANSITION_MIN_WINDOW_COVERAGE)

        extras = {
            "gt_skill": primary["skill"],
            "task_id": task_id,
            "episode_id": episode_id,
            "all_actions": overlapping,
            "window_coverage": round(total_coverage, 4),
            "is_transition": n_significant >= 2,  # noqa: PLR2004 — "≥2 actions" reads clearer inline
            "num_overlapping_actions": len(overlapping),
        }
        return primary["action_text"], extras
