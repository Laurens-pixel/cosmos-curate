# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""C-RADIOv4-H embedding stages for the video curation pipeline."""

import io
import time

import nvtx  # type: ignore[import-untyped]
import torch
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.models.cradio import CRadio
from cosmos_curate.pipelines.video.utils.data_model import (
    SplitPipeTask,
)
from cosmos_curate.pipelines.video.utils.decoder_utils import (
    FrameExtractionPolicy,
    FrameExtractionSignature,
    extract_frames,
)


class CRadioFrameCreationStage(CuratorStage):
    """Stage for creating C-RADIOv4-H input frames from video clips.

    Reads extracted frames from clips, samples and preprocesses them
    for the C-RADIOv4-H model. Runs on CPU only (no GPU needed).
    """

    def __init__(
        self,
        target_fps: float = 2.0,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the C-RADIOv4-H frame creation stage.

        Args:
            target_fps: Target frames per second for frame extraction.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._target_fps = target_fps
        self._frame_extraction_signature = FrameExtractionSignature(
            extraction_policy=FrameExtractionPolicy.sequence,
            target_fps=self._target_fps,
        ).to_str()
        self._model = CRadio(utils_only=True)
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def model(self) -> ModelInterface:
        """Get the C-RADIOv4-H model."""
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=1.0)

    @nvtx.annotate("CRadioFrameCreationStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process video data to create C-RADIOv4-H input frames.

        Args:
            tasks: Tasks containing video to process.

        Returns:
            Processed tasks with C-RADIOv4-H input frames.

        """
        max_fps: int = 20

        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            video.stage_timestamps["CRadioFrameCreationStage_start"] = time.time()
            for clip in video.clips:
                if clip.encoded_data is None:
                    clip.errors["encoded_data"] = "empty"
                    continue
                ef = clip.extracted_frames.resolve()
                if self._frame_extraction_signature not in ef:
                    clip.errors[f"frames-{self._frame_extraction_signature}"] = "missing"
                    logger.error(f"Clip {clip.uuid} has buffer but no extracted frames")
                    continue
                with self._timer.time_process():
                    frames = ef[self._frame_extraction_signature]
                    # Check if we need to re-extract at higher fps
                    target_num_frames = self._model.get_target_num_frames()
                    regen_fps = self._target_fps
                    while frames.shape[0] < target_num_frames:
                        regen_fps *= 2
                        if regen_fps > max_fps:
                            logger.error(f"Clip {clip.uuid} is too short to extract enough frames.")
                            break
                        if self._verbose:
                            logger.warning(
                                f"Clip {clip.uuid} has <{target_num_frames} frames. "
                                f"Re-extracting with higher target_fps={regen_fps}. "
                                f"Current # frames={frames.shape[0]}.",
                            )
                        with io.BytesIO(clip.encoded_data.resolve()) as fp:
                            frames = extract_frames(
                                fp,
                                extraction_policy=FrameExtractionPolicy.sequence,
                                sample_rate_fps=regen_fps,
                            )
                    # Create input frames for C-RADIOv4-H model, 432x432 and stack into a single tensor
                    clip.cradio_frames = self._model.formulate_input_frames(list(frames))
                # Done with extracted_frames
                clip.extracted_frames.drop()

            video.stage_timestamps["CRadioFrameCreationStage_end"] = time.time()

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks


class CRadioEmbeddingStage(CuratorStage):
    """Stage for generating embeddings from C-RADIOv4-H input frames.

    Processes preprocessed frames through C-RADIOv4-H with cpe_video_mode,
    mean-pools per-frame summaries into a single clip embedding.
    """

    def __init__(
        self,
        num_gpus_per_worker: float = 0.25,
        batch_size: int = 4,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the C-RADIOv4-H embedding stage.

        Args:
            num_gpus_per_worker: Number of GPUs per worker.
            batch_size: Batch size for processing.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._timer = StageTimer(self)
        self._num_gpus_per_worker = num_gpus_per_worker
        self._batch_size = batch_size
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = CRadio()
        self._process_count = 0

    def stage_setup(self) -> None:
        """Initialize stage resources and configuration."""
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
        self._model.setup()
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    @property
    def model(self) -> ModelInterface:
        """Get the C-RADIOv4-H model."""
        return self._model

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(gpus=self._num_gpus_per_worker)

    @nvtx.annotate("CRadioEmbeddingStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process video data to generate C-RADIOv4-H embeddings.

        Args:
            tasks: Tasks containing video to process.

        Returns:
            Processed tasks with generated embeddings.

        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            video.stage_timestamps["CRadioEmbeddingStage_start"] = time.time()
            with self._timer.time_process(len(video.clips)):
                for clip in video.clips:
                    if clip.cradio_frames is None:
                        clip.errors["cradio_frames"] = "empty"
                        continue
                    #Run the C-RADIOv4-H model on the input frames and mean pools all 8 
                    # frame embeddings into one embedding vector
                    embedding = self._model.encode_video_frames(clip.cradio_frames) 
                    if embedding.numel() == 0:
                        logger.error(f"Unable to compute C-RADIOv4-H embedding for clip={clip.uuid}")
                        clip.errors["cradio_embedding"] = "failed"
                    else:
                        clip.cradio_embedding = embedding.numpy()
                    # Done with cradio_frames
                    clip.cradio_frames = None

            video.stage_timestamps["CRadioEmbeddingStage_end"] = time.time()

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        # Free memory periodically
        self._process_count += 1
        if self._process_count % 10 == 0:
            torch.cuda.empty_cache()

        return tasks
