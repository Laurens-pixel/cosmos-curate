# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assembly101 coarse-annotation GT source.

Reads ``coarse_labels/{assembly_|disassembly_}_{seq}.txt`` files.  Each file
has tab-separated rows::

    start_frame  end_frame  action_label

where frame numbers are at **60 fps** (the HMC camera's native rate).

Video filenames in the dataset are ``human_assembly_NNNN_<cam>.mp4``.  The
seq is recovered from the manifest ``source_path`` field::

    source_path = "recordings/{seq}/{cam}.mp4"
    seq = source_path.split("/")[1]

The constructor takes:
* ``manifest_path`` — path to ``manifest.json`` (builds filename → seq map)
* ``labels_dir``    — path to ``coarse_labels/`` directory
"""

import json
import pathlib
from typing import Any

from loguru import logger

from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource

_TRANSITION_MIN_WINDOW_COVERAGE = 0.15


class Assembly101Gt(GtSource):
    """Look up GT coarse action labels from Assembly101 TSV annotation files."""

    def __init__(self, manifest_path: pathlib.Path, labels_dir: pathlib.Path) -> None:
        self._labels_dir = pathlib.Path(labels_dir)
        # video_filename → seq
        self._filename_to_seq: dict[str, str] = {}
        self._load_manifest(pathlib.Path(manifest_path))
        # seq → list of (start_frame, end_frame, action_label)
        self._seq_actions: dict[str, list[tuple[int, int, str]]] = {}
        self._loaded_seqs: set[str] = set()

    @staticmethod
    def name() -> str:
        return "assembly101"

    def _load_manifest(self, manifest_path: pathlib.Path) -> None:
        if not manifest_path.exists():
            logger.warning(f"Assembly101Gt: manifest not found: {manifest_path}")
            return
        try:
            entries = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(f"Assembly101Gt: failed to load manifest: {exc}")
            return
        for entry in entries:
            fname = entry.get("video_filename", "")
            src = entry.get("source_path", "")
            parts = src.split("/")
            if fname and len(parts) >= 2:
                self._filename_to_seq[fname] = parts[1]

    def _load_seq(self, seq: str) -> None:
        if seq in self._loaded_seqs:
            return
        self._loaded_seqs.add(seq)
        # try assembly_ then disassembly_ prefix
        for prefix in ("assembly_", "disassembly_"):
            path = self._labels_dir / f"{prefix}{seq}.txt"
            if not path.exists():
                continue
            actions: list[tuple[int, int, str]] = []
            try:
                for line in path.read_text().splitlines():
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    try:
                        sf, ef = int(parts[0]), int(parts[1])
                    except ValueError:
                        continue
                    label = parts[2].strip()
                    if label:
                        actions.append((sf, ef, label))
            except OSError as exc:
                logger.error(f"Assembly101Gt: failed to read {path}: {exc}")
                continue
            if actions:
                self._seq_actions[seq] = actions
                return
        logger.warning(f"Assembly101Gt: no coarse_labels file found for seq {seq!r}")

    def lookup(
        self,
        video_name: str,
        start_frame: int,
        end_frame: int,
    ) -> tuple[str, dict[str, Any]]:
        """Return majority-overlap GT action for a window (frames at 60 fps)."""
        fname = pathlib.Path(video_name).name
        seq = self._filename_to_seq.get(fname)
        if not seq:
            return "", {}

        self._load_seq(seq)
        actions = self._seq_actions.get(seq)
        if not actions:
            return "", {}

        window_len = max(1, end_frame - start_frame)
        overlapping: list[dict[str, Any]] = []
        for sf, ef, label in actions:
            overlap = max(0, min(ef, end_frame) - max(sf, start_frame))
            if overlap <= 0:
                continue
            action_len = max(1, ef - sf)
            overlapping.append(
                {
                    "action_text": label,
                    "start_frame": sf,
                    "end_frame": ef,
                    "overlap_frames": overlap,
                    "window_coverage": round(overlap / window_len, 4),
                    "action_coverage": round(overlap / action_len, 4),
                }
            )

        if not overlapping:
            return "", {}

        overlapping.sort(key=lambda a: a["overlap_frames"], reverse=True)
        primary = overlapping[0]
        total_coverage = min(1.0, sum(a["overlap_frames"] for a in overlapping) / window_len)
        n_significant = sum(1 for a in overlapping if a["window_coverage"] >= _TRANSITION_MIN_WINDOW_COVERAGE)

        extras = {
            "seq": seq,
            "all_actions": overlapping,
            "window_coverage": round(total_coverage, 4),
            "is_transition": n_significant >= 2,
            "num_overlapping_actions": len(overlapping),
        }
        return primary["action_text"], extras
