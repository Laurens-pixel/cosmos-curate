# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GT window provider interface and built-in implementations.

To add support for a new dataset, subclass ``GTWindowProvider`` and implement
``get_windows()``, then register it in ``_GT_PROVIDERS``.

Usage in the pipeline::

    provider = make_gt_window_provider("agibot", task_info_dir="/agibot_task_info")
    windows = provider.get_windows("/data/327_685046_head_color.mp4")
    # returns list[WindowFrameInfo] or None

"""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.utils.windowing_utils import WindowFrameInfo


# ── Abstract base ─────────────────────────────────────────────────────────────


class GTWindowProvider(ABC):
    """Return GT action frame ranges for a video.

    Implement this class for each dataset that provides ground-truth action
    boundaries.  The pipeline calls ``get_windows(video_path)`` for every video
    instead of running ``compute_windows()`` (TransNetV2-based windowing).
    """

    @abstractmethod
    def get_windows(self, video_path: str) -> list[WindowFrameInfo] | None:
        """Return GT windows for the video, or ``None`` if unavailable.

        Args:
            video_path: Absolute path (or basename) of the source video file.

        Returns:
            Ordered list of ``WindowFrameInfo(start, end)`` in source-video
            frame coordinates, or ``None`` when the video has no GT entry.

        """


# ── AgiBot implementation ─────────────────────────────────────────────────────

# Expected filename pattern: {task_id}_{episode_id}_{camera}.mp4
_AGIBOT_RE = re.compile(r"^(\d+)_(\d+)_")


class AgiBotGTWindowProvider(GTWindowProvider):
    """GT window provider for AgiBotWorld-Alpha.

    Reads ``task_<id>.json`` files from a task_info directory.  Each JSON is a
    list of episode entries whose ``label_info.action_config`` contains
    ``start_frame`` / ``end_frame`` boundaries for each labelled action.

    Video filenames must follow the pattern ``{task_id}_{episode_id}_{camera}.mp4``
    (e.g. ``327_685046_head_color.mp4``).

    Args:
        task_info_dir: Directory containing ``task_<id>.json`` files.

    """

    def __init__(self, task_info_dir: str | Path) -> None:
        """Load GT windows from all task JSON files."""
        self._windows: dict[tuple[int, int], list[WindowFrameInfo]] = {}
        self._load(Path(task_info_dir))

    def _load(self, task_info_dir: Path) -> None:
        json_files = sorted(task_info_dir.glob("task_*.json"))
        if not json_files:
            logger.warning(f"AgiBotGTWindowProvider: no task_*.json files found in {task_info_dir}")
            return

        for json_file in json_files:
            try:
                task_id = int(json_file.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                logger.warning(f"AgiBotGTWindowProvider: skipping {json_file.name} (cannot parse task_id)")
                continue

            for ep in json.loads(json_file.read_text()):
                episode_id = int(ep["episode_id"])
                actions = ep.get("label_info", {}).get("action_config", [])
                windows = [
                    WindowFrameInfo(start=a["start_frame"], end=a["end_frame"])
                    for a in actions
                    if "start_frame" in a and "end_frame" in a
                ]
                if windows:
                    self._windows[(task_id, episode_id)] = windows

        logger.info(
            f"AgiBotGTWindowProvider: loaded {len(self._windows)} episodes "
            f"from {len(json_files)} task file(s) in {task_info_dir}"
        )

    def get_windows(self, video_path: str) -> list[WindowFrameInfo] | None:
        """Return GT action windows for an AgiBot video.

        Args:
            video_path: Path whose stem starts with ``{task_id}_{episode_id}_``.

        Returns:
            Ordered list of ``WindowFrameInfo`` per GT action, or ``None``.

        """
        stem = Path(video_path).stem
        m = _AGIBOT_RE.match(stem)
        if not m:
            logger.warning(
                f"AgiBotGTWindowProvider: cannot parse task_id/episode_id from '{stem}'. "
                "Expected '{task_id}_{episode_id}_...'."
            )
            return None
        key = (int(m.group(1)), int(m.group(2)))
        windows = self._windows.get(key)
        if windows is None:
            logger.warning(f"AgiBotGTWindowProvider: no GT windows for episode {key} (file: {stem})")
        return windows


# ── Registry & factory ────────────────────────────────────────────────────────

_GT_PROVIDERS: dict[str, type[GTWindowProvider]] = {
    "agibot": AgiBotGTWindowProvider,
    # Add new dataset providers here, e.g.:
    # "nuscenes": NuScenesGTWindowProvider,
}


def list_gt_window_sources() -> list[str]:
    """Return the sorted list of registered GT window source names."""
    return sorted(_GT_PROVIDERS.keys())


def make_gt_window_provider(source: str, **kwargs: Any) -> GTWindowProvider:
    """Instantiate the GT window provider for ``source``.

    Args:
        source: Registered source name (e.g. ``"agibot"``).
        **kwargs: Constructor keyword arguments for the provider class.

    Returns:
        A ``GTWindowProvider`` instance.

    Raises:
        ValueError: If ``source`` is not registered.

    """
    if source not in _GT_PROVIDERS:
        msg = (
            f"Unknown GT window source: {source!r}. "
            f"Registered sources: {list_gt_window_sources()}"
        )
        raise ValueError(msg)
    return _GT_PROVIDERS[source](**kwargs)
