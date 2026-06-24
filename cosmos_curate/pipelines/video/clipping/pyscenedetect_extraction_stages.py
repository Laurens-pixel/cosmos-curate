# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scene extraction stage using PySceneDetect ContentDetector."""

import time
import uuid
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import nvtx  # type: ignore[import-untyped]
from loguru import logger  # type: ignore[import-not-found]

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.config.operation_context import make_pipeline_named_temporary_file
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.clipping.transnetv2_extraction_stages import _get_filtered_scenes
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask


class PySceneDetectClipExtractionStage(CuratorStage):
    """Stage for extracting clip spans using PySceneDetect's ContentDetector."""

    def __init__(  # noqa: PLR0913
        self,
        threshold: float = 27.0,
        min_scene_len_frames: int = 15,
        min_length_s: float | None = 2.0,
        min_length_frames: int | None = 48,
        max_length_s: float | None = 60.0,
        max_length_mode: Literal["truncate", "stride"] = "stride",
        crop_s: float | None = 0.5,
        *,
        entire_scene_as_clip: bool = True,
        num_cpus_per_worker: float = 2.0,
        limit_clips: int = 0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        self._timer = StageTimer(self)
        self.threshold = threshold
        self.min_scene_len_frames = min_scene_len_frames
        self.min_length_s = min_length_s
        self.min_length_frames = min_length_frames
        self.max_length_s = max_length_s
        self.max_length_mode: Literal["truncate", "stride"] = max_length_mode
        self.crop_s = crop_s
        self.entire_scene_as_clip = entire_scene_as_clip
        self._num_cpus_per_worker = num_cpus_per_worker
        self._limit_clips = limit_clips
        self._verbose = verbose
        self._log_stats = log_stats
        self._scenedetect = None
        self._scene_manager_cls = None
        self._content_detector_cls = None

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=self._num_cpus_per_worker)

    def stage_setup(self) -> None:
        """Lazy-import PySceneDetect to avoid import cost when not used."""
        extra_path = os.environ.get("PYTHONPATH", "")
        for part in extra_path.split(":"):
            if part and part not in sys.path and Path(part).is_dir():
                sys.path.insert(0, part)
        try:
            from scenedetect import SceneManager, open_video  # type: ignore[import-not-found]
            from scenedetect.detectors import ContentDetector  # type: ignore[import-not-found]
        except ImportError as e:
            msg = (
                "PySceneDetect is required for --shot-boundary-model pyscenedetect. "
                "Install with `pip install scenedetect[opencv]` in the runtime environment."
            )
            raise RuntimeError(msg) from e
        self._scenedetect = open_video
        self._scene_manager_cls = SceneManager
        self._content_detector_cls = ContentDetector

    def _get_min_length(self, framerate: float) -> int | None:
        min_length = int(np.ceil(self.min_length_s * framerate)) if self.min_length_s is not None else None
        if self.min_length_frames is not None:
            min_length = max(min_length, self.min_length_frames) if min_length is not None else self.min_length_frames
        return min_length

    def _get_max_length(self, framerate: float) -> int | None:
        return int(np.ceil(self.max_length_s * framerate)) if self.max_length_s is not None else None

    def _scene_list_to_spans(
        self, scene_list: list[tuple[object, object]], *, total_frames: int
    ) -> npt.NDArray[np.int32]:
        spans: list[tuple[int, int]] = []
        for start_tc, end_tc in scene_list:
            start_f = int(start_tc.get_frames())
            end_f = int(end_tc.get_frames())
            if end_f > start_f:
                spans.append((start_f, end_f))
        if not spans and self.entire_scene_as_clip and total_frames > 0:
            spans = [(0, total_frames)]
        return np.array(spans, dtype=np.int32).reshape(-1, 2)

    @nvtx.annotate("PySceneDetectClipExtractionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            video.stage_timestamps["PySceneDetectClipExtractionStage_start"] = time.time()
            if not video.has_metadata():
                logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
                continue
            if video.encoded_data is None:
                logger.warning(f"Missing encoded data for {video.input_video}. Skipping...")
                video.errors["pyscenedetect"] = "missing_encoded_data"
                continue
            assert video.metadata.framerate is not None
            data = video.encoded_data.resolve()
            if data is None:
                logger.warning(f"Encoded data resolved to None for {video.input_video}. Skipping...")
                video.errors["pyscenedetect"] = "empty_encoded_data"
                continue
            with (
                self._timer.time_process(),
                make_pipeline_named_temporary_file(sub_dir="pyscenedetect_split") as video_path,
            ):
                with video_path.open("wb") as fp:
                    fp.write(data)
                open_video = self._scenedetect
                scene_manager_cls = self._scene_manager_cls
                detector_cls = self._content_detector_cls
                if open_video is None or scene_manager_cls is None or detector_cls is None:
                    msg = "PySceneDetect stage_setup() must run before process_data()."
                    raise RuntimeError(msg)
                video_obj = open_video(Path(video_path).as_posix())
                scene_manager = scene_manager_cls()
                scene_manager.add_detector(
                    detector_cls(threshold=self.threshold, min_scene_len=self.min_scene_len_frames)
                )
                scene_manager.detect_scenes(video_obj)
                raw_scenes = self._scene_list_to_spans(
                    scene_manager.get_scene_list(),
                    total_frames=video.metadata.num_frames or 0,
                )
                filtered_scenes = _get_filtered_scenes(
                    raw_scenes,
                    min_length=self._get_min_length(video.metadata.framerate),
                    max_length=self._get_max_length(video.metadata.framerate),
                    max_length_mode=self.max_length_mode,
                    crop_length=(int(self.crop_s * video.metadata.framerate) if self.crop_s else None),
                )
                if self._verbose:
                    logger.info(
                        f"{video.input_video} returned {raw_scenes.shape[0]} scenes, "
                        f"{filtered_scenes.shape[0]} after filtering"
                    )
                s3_file = video.input_video
                for start_event, end_event in filtered_scenes:
                    clip = Clip(
                        uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{s3_file}_{start_event}_{end_event}"),
                        source_video=str(s3_file),
                        span=(
                            float(start_event) / video.metadata.framerate,
                            float(end_event) / video.metadata.framerate,
                        ),
                    )
                    video.clips.append(clip)
                    if self._limit_clips > 0 and len(video.clips) >= self._limit_clips:
                        break
                if not video.clips:
                    logger.warning(f"No scene boundaries predicted for {video.input_video}.")
            # Drop any pre-extracted frames if a previous stage produced them.
            video.frame_array.drop()
            video.stage_timestamps["PySceneDetectClipExtractionStage_end"] = time.time()
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
