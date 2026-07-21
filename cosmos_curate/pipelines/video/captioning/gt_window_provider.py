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

import inspect
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

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GTWindowProvider":
        """Instantiate from a dataset config dict, passing only recognised kwargs."""
        params = inspect.signature(cls.__init__).parameters
        kwargs = {k: cfg[k] for k in params if k not in ("self",) and k in cfg}
        return cls(**kwargs)

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


# ── Assembly101 implementation ────────────────────────────────────────────────


class Assembly101GTWindowProvider(GTWindowProvider):
    """GT window provider for Assembly101 using coarse TSV annotation files.

    Reads ``coarse_labels/{assembly_|disassembly_}{seq}.txt`` files where each
    line is ``start_frame\\tend_frame\\taction_label`` at **60 fps**.

    The video filename → seq mapping is built from the dataset manifest
    (``source_path = "recordings/{seq}/{cam}.mp4"``).

    Args:
        manifest_path: Path to ``manifest.json``.
        labels_dir: Path to the ``coarse_labels/`` directory.

    """

    def __init__(self, manifest_path: str | Path, labels_dir: str | Path) -> None:
        self._labels_dir = Path(labels_dir)
        self._filename_to_seq: dict[str, str] = {}
        self._seq_windows: dict[str, list[WindowFrameInfo]] = {}
        self._loaded_seqs: set[str] = set()
        self._load_manifest(Path(manifest_path))

    def _load_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            logger.warning(f"Assembly101GTWindowProvider: manifest not found: {manifest_path}")
            return
        for entry in json.loads(manifest_path.read_text()):
            fname = entry.get("video_filename", "")
            src = entry.get("source_path", "")
            parts = src.split("/")
            if fname and len(parts) >= 2:
                self._filename_to_seq[fname] = parts[1]

    def _load_seq(self, seq: str) -> None:
        if seq in self._loaded_seqs:
            return
        self._loaded_seqs.add(seq)
        for prefix in ("assembly_", "disassembly_"):
            path = self._labels_dir / f"{prefix}{seq}.txt"
            if not path.exists():
                continue
            windows: list[WindowFrameInfo] = []
            for line in path.read_text().splitlines():
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    sf, ef = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                if parts[2].strip() and ef > sf:
                    windows.append(WindowFrameInfo(start=sf, end=ef))
            if windows:
                self._seq_windows[seq] = windows
                return
        logger.warning(f"Assembly101GTWindowProvider: no coarse_labels file for seq {seq!r}")

    def get_windows(self, video_path: str) -> list[WindowFrameInfo] | None:
        fname = Path(video_path).name
        seq = self._filename_to_seq.get(fname)
        if not seq:
            logger.warning(f"Assembly101GTWindowProvider: no seq mapping for {fname!r}")
            return None
        self._load_seq(seq)
        return self._seq_windows.get(seq)


# ── WGO implementation ────────────────────────────────────────────────────────


class WGOGTWindowProvider(GTWindowProvider):
    """GT window provider for WGO-Bench using episode manifest segments.

    The manifest entries contain ``segments: [{start_sec, end_sec, subtask}]``
    and per-episode ``metadata.fps``.  Frame boundaries are computed as
    ``round(start_sec * fps)``.

    Args:
        manifest_path: Path to the WGO episode manifest JSON.
        default_fps: Fallback fps when metadata is missing (default 30.0).

    """

    def __init__(self, manifest_path: str | Path, default_fps: float = 30.0) -> None:
        self._default_fps = default_fps
        self._by_stem: dict[str, dict] = {}
        for rec in json.loads(Path(manifest_path).read_text()):
            stem = Path(rec.get("video_filename", f"{rec['id']}.mp4")).stem
            meta = rec.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            self._by_stem[stem] = {
                "segments": rec.get("segments") or [],
                "fps": float(meta.get("fps") or default_fps),
            }

    def get_windows(self, video_path: str) -> list[WindowFrameInfo] | None:
        stem = Path(video_path).stem
        ep = self._by_stem.get(stem)
        if not ep:
            logger.warning(f"WGOGTWindowProvider: no entry for {stem!r}")
            return None
        fps = ep["fps"]
        windows = [
            WindowFrameInfo(
                start=round(seg["start_sec"] * fps),
                end=round(seg["end_sec"] * fps),
            )
            for seg in ep["segments"]
            if seg.get("end_sec", 0) > seg.get("start_sec", 0)
        ]
        return windows or None


# ── Registry & factory ────────────────────────────────────────────────────────

_GT_PROVIDERS: dict[str, type[GTWindowProvider]] = {
    "agibot": AgiBotGTWindowProvider,
    "assembly101": Assembly101GTWindowProvider,
    "wgo": WGOGTWindowProvider,
}


def list_gt_window_sources() -> list[str]:
    """Return the sorted list of registered GT window source names."""
    return sorted(_GT_PROVIDERS.keys())


def make_gt_window_provider(source: str, cfg: dict[str, Any]) -> GTWindowProvider:
    """Instantiate the GT window provider for ``source`` from a dataset config dict.

    Args:
        source: Registered source name (e.g. ``"agibot"``).
        cfg: Dataset config dict — the provider picks only the keys it needs.

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
    return _GT_PROVIDERS[source].from_config(cfg)
