# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WGO-Bench GT source (macrodata/WGO-Bench).

Reads ``episode_manifest.json`` produced by ``download_wgo.py``. Each episode has gold
``segments``: ``{start_sec, end_sec, subtask}``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource


class WgoBenchGt(GtSource):
    """Look up the subtask label with the most temporal overlap in a caption window."""

    def __init__(self, manifest_path: pathlib.Path, default_fps: float = 30.0) -> None:
        self._default_fps = default_fps
        records = json.loads(pathlib.Path(manifest_path).read_text())
        self._by_stem: dict[str, dict[str, Any]] = {}
        for rec in records:
            stem = pathlib.Path(rec.get("video_filename", f"{rec['id']}.mp4")).stem
            meta = rec.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            self._by_stem[stem] = {
                "instruction": rec.get("instruction", ""),
                "segments": rec.get("segments") or [],
                "fps": float(meta.get("fps") or default_fps),
            }

    @staticmethod
    def name() -> str:
        return "wgo"

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        stem = pathlib.Path(video_name).stem
        ep = self._by_stem.get(stem)
        if not ep:
            return "", {}

        fps = float(ep.get("fps") or self._default_fps)
        win_start = start_frame / fps
        win_end = end_frame / fps

        best_label = ""
        best_overlap = 0.0
        for seg in ep["segments"]:
            overlap = max(0.0, min(seg["end_sec"], win_end) - max(seg["start_sec"], win_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = str(seg.get("subtask", ""))

        if not best_label:
            return "", {}
        return best_label, {
            "instruction": ep.get("instruction", ""),
            "annotation_overlap_s": round(best_overlap, 3),
        }
