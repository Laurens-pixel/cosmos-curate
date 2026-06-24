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
"""Model-driven shot/action boundary extraction stage."""

import time
import uuid

import numpy as np
import nvtx  # type: ignore[import-untyped]
from loguru import logger  # type: ignore[import-not-found]

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.config.operation_context import make_pipeline_named_temporary_file
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.pipelines.video.clipping.shot_boundary_models import (
    DetectorConfig,
    build_boundary_detector,
)
from cosmos_curate.pipelines.video.clipping.transnetv2_extraction_stages import _get_filtered_scenes
from cosmos_curate.pipelines.video.utils.data_model import Clip, SplitPipeTask


class ModelBoundaryClipExtractionStage(CuratorStage):
    """Stage that extracts clip spans using a named boundary detector model."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        detector_config: DetectorConfig,
        min_length_s: float | None = 2.0,
        min_length_frames: int | None = 48,
        max_length_s: float | None = 60.0,
        max_length_mode: str = "stride",
        crop_s: float | None = 0.5,
        *,
        entire_scene_as_clip: bool = True,
        num_cpus_per_worker: float = 2.0,
        num_gpus_per_worker: float = 0.0,
        limit_clips: int = 0,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.model_name = model_name
        self.detector_config = detector_config
        self.min_length_s = min_length_s
        self.min_length_frames = min_length_frames
        self.max_length_s = max_length_s
        self.max_length_mode = max_length_mode
        self.crop_s = crop_s
        self.entire_scene_as_clip = entire_scene_as_clip
        self._num_cpus_per_worker = num_cpus_per_worker
        self._num_gpus_per_worker = num_gpus_per_worker
        self._limit_clips = limit_clips
        self._verbose = verbose
        self._log_stats = log_stats
        self._timer = StageTimer(self)
        self._detector = None

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(cpus=self._num_cpus_per_worker, gpus=self._num_gpus_per_worker)

    @property
    def conda_env_name(self) -> str:
        """Semantic and VLM boundary detectors need torch + transformers."""
        return "unified"

    def stage_setup(self) -> None:
        self._detector = build_boundary_detector(self.model_name, self.detector_config)

    def _get_min_length(self, framerate: float) -> int | None:
        min_length = int(np.ceil(self.min_length_s * framerate)) if self.min_length_s is not None else None
        if self.min_length_frames is not None:
            min_length = max(min_length, self.min_length_frames) if min_length is not None else self.min_length_frames
        return min_length

    def _get_max_length(self, framerate: float) -> int | None:
        return int(np.ceil(self.max_length_s * framerate)) if self.max_length_s is not None else None

    def _boundary_ts_to_scene_spans(self, boundaries_s: list[float], total_frames: int, fps: float) -> np.ndarray:
        if total_frames <= 0 or fps <= 0:
            return np.array([], dtype=np.int32).reshape(-1, 2)
        boundaries_f = sorted({max(0, min(int(t * fps), total_frames)) for t in boundaries_s})
        spans: list[tuple[int, int]] = []
        start = 0
        for b in boundaries_f:
            if b > start:
                spans.append((start, b))
                start = b
        if start < total_frames:
            spans.append((start, total_frames))
        if not spans and self.entire_scene_as_clip:
            spans = [(0, total_frames)]
        return np.array(spans, dtype=np.int32).reshape(-1, 2)

    @nvtx.annotate("ModelBoundaryClipExtractionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        detector = self._detector
        if detector is None:
            msg = "Detector not initialized; stage_setup() was not called."
            raise RuntimeError(msg)
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            video.stage_timestamps[f"{self.model_name}_start"] = time.time()
            if not video.has_metadata():
                logger.warning(f"Incomplete metadata for {video.input_video}. Skipping...")
                continue
            if video.encoded_data is None:
                logger.warning(f"Missing encoded data for {video.input_video}. Skipping...")
                video.errors["model_boundary"] = "missing_encoded_data"
                continue
            assert video.metadata.framerate is not None
            data = video.encoded_data.resolve()
            if data is None:
                logger.warning(f"Encoded data resolved to None for {video.input_video}. Skipping...")
                video.errors["model_boundary"] = "empty_encoded_data"
                continue
            with (
                self._timer.time_process(),
                make_pipeline_named_temporary_file(sub_dir="model_boundary_split") as video_path,
            ):
                with video_path.open("wb") as fp:
                    fp.write(data)
                boundary_ts = detector.detect_boundaries(str(video_path))
                raw_scenes = self._boundary_ts_to_scene_spans(
                    boundary_ts,
                    total_frames=video.metadata.num_frames or 0,
                    fps=video.metadata.framerate,
                )
                filtered_scenes = _get_filtered_scenes(
                    raw_scenes,
                    min_length=self._get_min_length(video.metadata.framerate),
                    max_length=self._get_max_length(video.metadata.framerate),
                    max_length_mode=self.max_length_mode,  # type: ignore[arg-type]
                    crop_length=(int(self.crop_s * video.metadata.framerate) if self.crop_s else None),
                )
                if filtered_scenes.shape[0] == 0 and self.entire_scene_as_clip:
                    total_frames = video.metadata.num_frames or 0
                    if total_frames > 0:
                        fallback = np.array([[0, total_frames]], dtype=np.int32)
                        filtered_scenes = _get_filtered_scenes(
                            fallback,
                            min_length=self._get_min_length(video.metadata.framerate),
                            max_length=self._get_max_length(video.metadata.framerate),
                            max_length_mode=self.max_length_mode,  # type: ignore[arg-type]
                            crop_length=None,
                        )
                if self._verbose:
                    logger.info(
                        f"{video.input_video} detector={self.model_name} returned "
                        f"{len(boundary_ts)} boundaries, {filtered_scenes.shape[0]} scenes after filtering"
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
                    logger.warning(f"No boundaries predicted for {video.input_video} with {self.model_name}.")
            video.frame_array.drop()
            video.stage_timestamps[f"{self.model_name}_end"] = time.time()
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks
