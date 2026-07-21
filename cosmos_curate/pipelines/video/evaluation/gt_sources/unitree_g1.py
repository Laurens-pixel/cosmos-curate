# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 GT source.

Each task directory contains a ``manifest.json`` with entries like::

    {"video_filename": "..._ep000000.mp4", "task": "take plates into dishwasher", ...}

The ``task`` field is the only GT label — there are no per-frame annotations.
``lookup()`` returns the task text covering the entire video, regardless of
the queried frame range.

The constructor takes the base ``ego_datasets/`` directory under which
``unitreerobotics__G1_*/manifest.json`` files live.
"""

import json
import pathlib
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource


class UnitreeG1Gt(GtSource):
    """GT source for Unitree G1 robot datasets — task label per episode."""

    def __init__(self, base_dir: pathlib.Path) -> None:
        self._filename_to_task: dict[str, str] = {}
        self._load_manifests(pathlib.Path(base_dir))

    @staticmethod
    def name() -> str:
        return "unitree_g1"

    def _load_manifests(self, base_dir: pathlib.Path) -> None:
        if not base_dir.exists():
            logger.warning(f"UnitreeG1Gt: base_dir not found: {base_dir}")
            return
        manifests = list(base_dir.glob("unitreerobotics__G1_*/manifest.json"))
        if not manifests:
            logger.warning(f"UnitreeG1Gt: no unitreerobotics__G1_*/manifest.json found under {base_dir}")
            return
        for mpath in manifests:
            try:
                entries = json.loads(mpath.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(f"UnitreeG1Gt: failed to load {mpath}: {exc}")
                continue
            for entry in entries:
                fname = entry.get("video_filename", "")
                task = entry.get("task", "")
                if fname and task:
                    self._filename_to_task[fname] = task

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return the task label for the episode (whole-video GT, no per-frame info)."""
        fname = pathlib.Path(video_name).name
        task = self._filename_to_task.get(fname, "")
        if not task:
            return "", {}
        extras: dict[str, Any] = {
            "window_coverage": 1.0,
            "is_transition": False,
            "num_overlapping_actions": 1,
            "all_actions": [
                {
                    "action_text": task,
                    "start_frame": 0,
                    "end_frame": end_frame,
                    "overlap_frames": end_frame - start_frame,
                    "window_coverage": 1.0,
                    "action_coverage": 1.0,
                }
            ],
        }
        return task, extras
