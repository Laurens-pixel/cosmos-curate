# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert curated clips + captions + source-dataset actions into a ``LeRobotDataset``.

LeRobot (https://github.com/huggingface/lerobot) is the open-source stack we target: it
provides the dataset format, a language-conditioned diffusion policy, and training utilities,
and it directly accepts the (observation, action, language) structure our curation produces
once joined with the source dataset's action/state streams.

The cosmos-curate pipeline supplies, per clip: the video frames (clip mp4 / per-window frames)
and a language instruction (the generated caption). The *actions* and *proprioceptive state*
come from the source robot dataset (LIBERO / DROID / AgiBotWorld), re-joined by episode id.
This module bundles both into :class:`CuratedEpisode` objects and writes a ``LeRobotDataset``.

The LeRobot import is lazy so the rest of the suite runs without it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass
class CuratedEpisode:
    """One language-conditioned demonstration ready for LeRobot conversion.

    Attributes:
        frames: ``(T, H, W, 3)`` uint8 RGB observations (from the curated clip).
        actions: ``(T, action_dim)`` action vectors (from the source dataset).
        states: ``(T, state_dim)`` proprioceptive state (from the source dataset).
        instruction: Language instruction (the curated caption).
        fps: Frame rate used for the dataset timeline.

    """

    frames: npt.NDArray[np.uint8]
    actions: FloatArray
    states: FloatArray
    instruction: str
    fps: float = 10.0

    def validate(self) -> None:
        """Check that frames/actions/states share the same time dimension."""
        t = len(self.frames)
        if not (len(self.actions) == len(self.states) == t):
            msg = f"length mismatch: frames={t} actions={len(self.actions)} states={len(self.states)}"
            raise ValueError(msg)
        if t == 0:
            msg = "CuratedEpisode is empty"
            raise ValueError(msg)


def build_lerobot_dataset(
    episodes: list[CuratedEpisode],
    repo_id: str,
    root: str | Path,
    robot_type: str = "panda",
):  # noqa: ANN201 - return type is lerobot.LeRobotDataset (optional dep)
    """Write ``episodes`` to a ``LeRobotDataset`` on disk and return it.

    Mirrors the current LeRobot ``create`` / ``add_frame`` / ``save_episode`` API. Each frame
    carries the language instruction in the ``task`` field, which LeRobot uses for
    language-conditioned policies.

    Raises:
        RuntimeError: If LeRobot is not installed.

    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - needs the optional lerobot stack
        msg = "build_lerobot_dataset requires 'lerobot' (pip install lerobot)."
        raise RuntimeError(msg) from exc

    if not episodes:
        msg = "No episodes to convert."
        raise ValueError(msg)
    for ep in episodes:
        ep.validate()

    first = episodes[0]
    height, width = first.frames.shape[1:3]
    features = {
        "observation.image": {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (first.states.shape[1],), "names": None},
        "action": {"dtype": "float32", "shape": (first.actions.shape[1],), "names": None},
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(first.fps)),
        root=Path(root),
        robot_type=robot_type,
        features=features,
        use_videos=True,
    )

    for ep in episodes:
        for frame, state, action in zip(ep.frames, ep.states, ep.actions, strict=True):
            dataset.add_frame(
                {
                    "observation.image": np.asarray(frame, dtype=np.uint8),
                    "observation.state": np.asarray(state, dtype=np.float32),
                    "action": np.asarray(action, dtype=np.float32),
                },
                task=ep.instruction,
            )
        dataset.save_episode()

    return dataset
