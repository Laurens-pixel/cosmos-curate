# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny cosmos-curate-style output tree + ground truth for smoke testing.

Produces the same on-disk layout the pipeline writes (``metas/v0/*.json``,
``v0/all_window_captions.json``, ``v0/all_window_judgments.json``, ``cradio_embd/*.pickle``)
for two illustrative source videos:

- an AgiBot-like manipulation video with labelled sub-task segments, and
- a YouCook2-like cooking video with labelled recipe steps.

Embeddings are constructed so that a clip and its caption share a label-anchored vector
(plus noise), making retrieval/grounding metrics produce meaningful, non-degenerate values.
A parallel :class:`MockGroundTruth` object is returned for the evaluators to score against.
"""

import json
import pickle
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from downstream_eval.common.types import Segment

_EMBED_DIM = 32


@dataclass
class MockClip:
    """A single predicted clip in the mock dataset."""

    uuid: str
    source_video: str
    start: float
    end: float
    framerate: float
    caption: str
    references: list[str]
    label: str
    judge_verdict: str
    human_correct: bool | None
    embedding: np.ndarray


@dataclass
class MockGroundTruth:
    """Ground truth + predictions bundled for the evaluators."""

    output_dir: Path
    gt_segments: dict[str, list[Segment]] = field(default_factory=dict)
    gt_action_intervals: dict[str, list[tuple[float, float, str]]] = field(default_factory=dict)
    clips: list[MockClip] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)


def _label_vector(label: str, rng: np.random.Generator) -> np.ndarray:
    seed = abs(hash(label)) % (2**32)
    base = np.random.default_rng(seed).standard_normal(_EMBED_DIM).astype(np.float32)
    return base + 0.15 * rng.standard_normal(_EMBED_DIM).astype(np.float32)


_AGIBOT = [
    ("reach for the red block", "a robot arm reaches toward the red block on the table"),
    ("grasp the red block", "the gripper closes and grasps the red block"),
    ("lift the red block", "the arm lifts the red block off the table surface"),
    ("place the block in the bin", "the robot places the red block into the blue bin"),
]
_YOUCOOK = [
    ("chop the onions", "a person chops the onions on a cutting board"),
    ("heat oil in the pan", "oil is heated in a frying pan on the stove"),
    ("add the onions to the pan", "the chopped onions are added to the hot pan"),
    ("stir the mixture", "the cook stirs the mixture with a wooden spoon"),
]


def _build_video(
    source_video: str,
    steps: list[tuple[str, str]],
    rng: np.random.Generator,
    *,
    duration_per_step: float,
    add_over_segmentation: bool,
) -> tuple[list[Segment], list[tuple[float, float, str]], list[MockClip]]:
    framerate = 30.0
    gt_segments: list[Segment] = []
    gt_intervals: list[tuple[float, float, str]] = []
    clips: list[MockClip] = []

    cursor = 0.0
    for idx, (label, caption) in enumerate(steps):
        gt_start = cursor
        gt_end = cursor + duration_per_step
        gt_segments.append(Segment(start=gt_start, end=gt_end, label=label))
        gt_intervals.append((gt_start, gt_end, label))

        # Predicted boundaries are slightly off from GT (jitter ~0.2s).
        pred_start = gt_start + float(rng.uniform(-0.2, 0.2)) if idx > 0 else 0.0
        pred_end = gt_end + float(rng.uniform(-0.2, 0.2)) if idx < len(steps) - 1 else gt_end

        verdict = "CORRECT" if rng.random() > 0.2 else "INCORRECT"
        human_correct = None
        if idx % 2 == 0:  # human label available for half the windows
            human_correct = verdict == "CORRECT"

        clips.append(
            MockClip(
                uuid=str(uuid.uuid4()),
                source_video=source_video,
                start=round(pred_start, 3),
                end=round(pred_end, 3),
                framerate=framerate,
                caption=caption,
                references=[label, caption],
                label=label,
                judge_verdict=verdict,
                human_correct=human_correct,
                embedding=_label_vector(label, rng),
            )
        )
        cursor = gt_end

    if add_over_segmentation:
        # Split the last GT segment into two predicted clips to create over-segmentation.
        last = clips[-1]
        mid = (last.start + last.end) / 2
        extra = MockClip(
            uuid=str(uuid.uuid4()),
            source_video=source_video,
            start=round(mid, 3),
            end=last.end,
            framerate=framerate,
            caption=last.caption,
            references=last.references,
            label=last.label,
            judge_verdict="CORRECT",
            human_correct=None,
            embedding=_label_vector(last.label, rng),
        )
        last.end = round(mid, 3)
        clips.append(extra)

    return gt_segments, gt_intervals, clips


def _write_pipeline_outputs(gt: MockGroundTruth) -> None:
    out = gt.output_dir
    (out / "metas" / "v0").mkdir(parents=True, exist_ok=True)
    (out / "v0").mkdir(parents=True, exist_ok=True)
    (out / "cradio_embd").mkdir(parents=True, exist_ok=True)

    all_captions: dict[str, dict[str, dict[str, str]]] = {}
    all_judgments: dict[str, dict[str, dict[str, dict[str, object]]]] = {}

    for clip in gt.clips:
        start_frame = int(round(clip.start * clip.framerate))
        end_frame = int(round(clip.end * clip.framerate))
        window_key = f"{start_frame}_{end_frame}"
        judge_record: dict[str, object] = {
            "verdict": clip.judge_verdict,
            "score": 1 if clip.judge_verdict == "CORRECT" else 0,
            "explanation": "mock",
            "gt_action_text": clip.label,
            "prompt_variant": "lenient_binary",
        }
        if clip.human_correct is not None:
            judge_record["gt_extras"] = {"human_correct": clip.human_correct}

        meta = {
            "span_uuid": clip.uuid,
            "source_video": clip.source_video,
            "duration_span": [clip.start, clip.end],
            "framerate_source": clip.framerate,
            "framerate": clip.framerate,
            "windows": [
                {
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "qwen_caption": clip.caption,
                    "judge": {"lenient_binary": judge_record},
                }
            ],
            "valid": True,
        }
        (out / "metas" / "v0" / f"{clip.uuid}.json").write_text(json.dumps(meta, indent=2))

        with (out / "cradio_embd" / f"{clip.uuid}.pickle").open("wb") as fh:
            pickle.dump(clip.embedding, fh)

        all_captions.setdefault(clip.source_video, {})[clip.uuid] = {window_key: clip.caption}
        all_judgments.setdefault(clip.source_video, {})[clip.uuid] = {
            window_key: {"lenient_binary": judge_record}
        }

    (out / "v0" / "all_window_captions.json").write_text(json.dumps(all_captions, indent=2))
    (out / "v0" / "all_window_judgments.json").write_text(json.dumps(all_judgments, indent=2))


def build_mock_dataset(root: str | Path, seed: int = 0) -> MockGroundTruth:
    """Build the mock output tree under ``root`` and return its ground truth."""
    rng = np.random.default_rng(seed)
    gt = MockGroundTruth(output_dir=Path(root))

    agibot_segs, agibot_intervals, agibot_clips = _build_video(
        "agibot/episode_0.mp4", _AGIBOT, rng, duration_per_step=3.0, add_over_segmentation=True
    )
    youcook_segs, youcook_intervals, youcook_clips = _build_video(
        "youcook2/recipe_0.mp4", _YOUCOOK, rng, duration_per_step=5.0, add_over_segmentation=False
    )

    gt.gt_segments = {
        "agibot/episode_0.mp4": agibot_segs,
        "youcook2/recipe_0.mp4": youcook_segs,
    }
    gt.gt_action_intervals = {
        "agibot/episode_0.mp4": agibot_intervals,
        "youcook2/recipe_0.mp4": youcook_intervals,
    }
    gt.clips = agibot_clips + youcook_clips
    gt.class_names = sorted({c.label for c in gt.clips})

    _write_pipeline_outputs(gt)
    return gt
