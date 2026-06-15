# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for loaders and the mock-data round trip."""

from pathlib import Path

from downstream_eval.common.io import (
    load_clip_embeddings,
    load_clip_records,
    load_window_captions,
    load_window_judgments,
    predicted_segments_by_video,
)
from downstream_eval.mock_data import build_mock_dataset


def test_mock_dataset_round_trips(tmp_path: Path) -> None:
    gt = build_mock_dataset(tmp_path, seed=3)

    clips = load_clip_records(gt.output_dir)
    assert len(clips) == len(gt.clips)
    assert all(c.span[1] >= c.span[0] for c in clips)
    assert all(c.windows for c in clips)

    captions = load_window_captions(gt.output_dir)
    assert set(captions) == set(gt.gt_segments)

    judgments = load_window_judgments(gt.output_dir)
    assert judgments

    embeddings = load_clip_embeddings(gt.output_dir, algorithm="cradio")
    assert len(embeddings) == len(gt.clips)
    assert all(vec.ndim == 1 for vec in embeddings.values())


def test_predicted_segments_grouped_and_sorted(tmp_path: Path) -> None:
    gt = build_mock_dataset(tmp_path, seed=1)
    clips = load_clip_records(gt.output_dir)
    by_video = predicted_segments_by_video(clips)
    assert set(by_video) == set(gt.gt_segments)
    for segments in by_video.values():
        starts = [s.start for s in segments]
        assert starts == sorted(starts)
