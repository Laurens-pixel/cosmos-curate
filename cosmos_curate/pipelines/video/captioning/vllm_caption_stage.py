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
"""vLLM caption stages.

The VllmPrepStage and VllmCaptionStage classes are designed to be used
in any pipeline. Because they are designed to be used in any pipeline, they
are generic and not specific to any particular pipeline or task type.

For the VllmPrepStage and VllmCaptionStage to function properly, the
the tasks must have these attributes/methods:

- video: The video to process.
- stage_perf: A dictionary to store performance statistics.
- get_major_size: A method to get the major size of the task.

"""

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import nvtx  # type: ignore[import-untyped]
import tenacity
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource, PipelineTask
from cosmos_curate.core.utils.infra.gpu_start_helper import (
    gpu_stage_cleanup,
    gpu_stage_startup,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.core.utils.model import conda_utils, model_utils
from cosmos_curate.models.all_models import get_all_models_by_id
from cosmos_curate.models.prompts import get_prompt, get_stage2_prompt
from cosmos_curate.models.vllm_model_ids import get_vllm_model_id
from cosmos_curate.pipelines.video.captioning.gt_window_provider import GTWindowProvider, make_gt_window_provider
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import (
    Video,
    VllmConfig,
    Window,
    WindowConfig,
    get_video_from_task,
)

if conda_utils.is_running_in_env("unified"):
    if TYPE_CHECKING:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

    from cosmos_curate.models.vllm_interface import (
        auto_processor,
        make_metadata,
        make_model_inputs,
        make_multiview_model_input,
        sampling_params,
        vllm_caption,
        vllm_model,
    )

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)  # Suppress warnings and info from vLLM


T = TypeVar("T", bound=PipelineTask)


def _get_windows_from_tasks[T: PipelineTask](tasks: list[T]) -> tuple[list[Window], list[str]]:
    """Get the windows from a list of tasks.

    Args:
        tasks: The tasks with video -> clips -> windows.

    Returns:
        The windows and clip uuids from the task.

    Raises:
        TypeError: If the task does not have a video attribute.

    """
    windows: list[Window] = []
    clip_uuids: list[str] = []
    for task in tasks:
        video = get_video_from_task(task)
        for clip in video.clips:
            if not clip.windows:
                logger.warning(f"Clip {clip.uuid} has no windows")
                clip.errors["clip_windowing"] = "empty"
                continue
            windows += clip.windows
            clip_uuids += [str(clip.uuid)] * len(clip.windows)

    return windows, clip_uuids


def _get_stage2_prompts(vllm_config: VllmConfig, num_windows: int) -> list[str | None]:
    """Get the stage 2 prompts for the vLLM model.

    Args:
        vllm_config: The configuration for the vLLM model.
        num_windows: The number of windows to get the stage 2 prompts for.

    Returns:
        The stage 2 prompts for the vLLM model.

    """
    if vllm_config.stage2_caption:
        return [get_stage2_prompt(vllm_config.stage2_prompt_text)] * num_windows
    return [None] * num_windows


def _scatter_captions(
    windows: list[Window], captions: list[str], clip_uuids: list[str], model_variant: str, *, verbose: bool
) -> None:
    """Scatter the captions back to the windows.

    Args:
        windows: The windows to scatter the captions to.
        captions: The captions to scatter.
        clip_uuids: The clip uuids to scatter the captions to.
        model_variant: The variant of the model.
        verbose: Whether to print verbose logs.

    """
    for window, caption, clip_uuid in zip(windows, captions, clip_uuids, strict=True):
        window.caption[model_variant] = caption                 #Sets the caption for the window
        if verbose:
            logger.info(f"Caption for clip {clip_uuid}: {caption}")


def _scatter_reasoning(windows: list[Window], reasonings: list[str | None], model_variant: str) -> None:
    """Scatter reasoning traces back to the windows (reasoning models only).

    Args:
        windows: The windows to scatter the reasoning to.
        reasonings: The reasoning traces, aligned with ``windows``. Entries are None for
            models that do not emit a separate reasoning trace, in which case nothing is stored.
        model_variant: The variant of the model.

    """
    for window, reasoning in zip(windows, reasonings, strict=True):
        if reasoning:
            window.reasoning[model_variant] = reasoning


def _free_vllm_inputs(windows: list[Window], model_variant: str, *, keep_mp4: bool = False) -> None:
    """Free unused memory for the model variant.

    Args:
        windows: The windows to free unused memory for.
        model_variant: The variant of the model.
        keep_mp4: Whether to keep the mp4 bytes.

    """
    for window in windows:
        window.model_input.pop(model_variant, None)
        if not keep_mp4:
            window.mp4_bytes.drop()


class VllmModelInterface(ModelInterface):
    """Information about a vLLM model."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        """Initialize the vLLM model interface."""
        self._vllm_config = vllm_config

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name."""
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names."""
        model_variant = get_vllm_model_id(self._vllm_config.model_variant)
        models = get_all_models_by_id()
        model = models.get(model_variant)

        if model is None:
            msg = f"Model not found for {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)

        model_id = model.get("model_id")
        if model_id is None:
            msg = f"Model ID not found for variant {self._vllm_config.model_variant} -> {model_variant}"
            raise ValueError(msg)

        return [cast("str", model_id)]

    def setup(self) -> None:
        """Set up the vLLM model interface."""


class VllmPrepStage(CuratorStage):
    """Stage that prepares cosmos-curate video data for vLLM multimodal model processing."""

    # Camera names expected in multi-view mode, in input order (left, front, right).
    _MULTIVIEW_CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    # Regex to parse nuScenes-style filenames: scene-XXXX_CAM_FRONT.mp4
    _SCENE_CAMERA_RE = re.compile(r"^(.+?)_(CAM_[A-Z_]+)\.mp4$", re.IGNORECASE)

    def __init__(
        self,
        vllm_config: VllmConfig,
        window_config: WindowConfig,
        *,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
        multi_view: bool = False,
        gt_window_source: str | None = None,
        gt_window_cfg: dict | None = None,
    ) -> None:
        """Initialize the vLLM Preparation Stage.

        Args:
            vllm_config: Configuration for the vLLM model.
            window_config: Configuration for the windowing.
            keep_mp4: Keep mp4 bytes for the clips in memory.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.
            multi_view: If True, group camera triplets per scene and produce one
                combined multi-view caption per scene (stored on CAM_FRONT clip only).
            gt_window_source: GT window provider name (e.g. ``"agibot"``).  When set,
                ``compute_windows`` is bypassed and GT action frame ranges are used instead.
            gt_window_cfg: Dataset config dict passed to the provider's ``from_config()``.

        """
        super().__init__()

        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._window_config = window_config
        self._verbose = verbose
        self._log_stats = log_stats
        self._processor: AutoProcessor | None = None
        self._keep_mp4 = keep_mp4
        self._model = VllmModelInterface(self._vllm_config)
        self._multi_view = multi_view
        self._gt_window_source = gt_window_source
        self._gt_window_cfg = gt_window_cfg or {}
        self._gt_provider: GTWindowProvider | None = None
        # Buffer: scene_id -> {camera_name -> (Video, task)}. Used only when multi_view=True.
        # Tasks are stored here so they can be retrieved across batches when a triplet completes.
        self._scene_buffer: dict[str, dict[str, tuple[Video, Any]]] = {}

    def secondary_name(self) -> str:
        """Get the secondary name of the stage.

        Returns:
            The secondary name of the stage.

        """
        # mypy is not smart enough to know that self._vllm_config.model_variant is a str
        # but mypy also thinks that this is a redundant cast
        return cast("str", self._vllm_config.model_variant)  # type: ignore[redundant-cast]

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        if self._multi_view:
            # Force exactly 1 worker by requesting >half of the schedulable CPUs.
            # cosmos_xenna allocates cpu_allocation_percentage=0.95 of total CPUs.
            # Requesting schedulable//2 + 1 guarantees floor(schedulable / request) == 1
            # while staying within the schedulable budget (avoids the allocation panic).
            try:
                import ray as _ray  # noqa: PLC0415
                total_ray_cpus = int(_ray.cluster_resources().get("CPU", 72)) if _ray.is_initialized() else 72
            except Exception:  # noqa: BLE001
                total_ray_cpus = 72
            schedulable = int(total_ray_cpus * 0.95)
            cpus = max(schedulable // 2 + 1, 1)
            return CuratorStageResource(cpus=cpus)

        #Default number of CPUs for the vLLM model: 2.0
        return CuratorStageResource(cpus=self._vllm_config.num_cpus_for_prepare)

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        self._processor = auto_processor(self._vllm_config)
        if self._gt_window_source is not None:
            self._gt_provider = make_gt_window_provider(self._gt_window_source, self._gt_window_cfg)

    def _parse_scene_camera(self, video: Video) -> tuple[str, str]:
        """Parse the source video filename into (scene_id, camera_name).

        Expects filenames like ``scene-0061_CAM_FRONT.mp4``.

        Args:
            video: The video whose source path to parse.

        Returns:
            A tuple of (scene_id, camera_name). If the filename does not match
            the expected pattern, returns (stem, "unknown") and logs a warning.

        """
        filename = Path(video.input_path).name
        match = self._SCENE_CAMERA_RE.match(filename)
        if match:
            return match.group(1), match.group(2).upper()
        logger.warning(
            f"Multi-view: could not parse scene/camera from '{filename}'. "
            "Expected pattern '<scene_id>_CAM_<NAME>.mp4'. Treating as standalone."
        )
        return Path(filename).stem, "unknown"

    def _prep_multiview_windows(self, group: dict[str, tuple[Video, Any]], prompt: str) -> None:
        """Prepare multi-view windows for a complete camera triplet.

        Extracts frames from CAM_FRONT_LEFT, CAM_FRONT, and CAM_FRONT_RIGHT,
        then creates a single combined LLM input (stored on the CAM_FRONT clip
        windows only). Left and right clips have their windows created but receive
        no model_input, so VllmCaptionStage will skip them.

        Args:
            group: Mapping of camera_name -> (Video, task) for one scene (must contain
                   all three cameras in _MULTIVIEW_CAMERAS).
            prompt: The captioning prompt to use.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        num_decode_threads = max(1, int(self.resources.cpus) + 1)

        # When combining 3 camera videos in one vLLM request, the total visual
        # token count is 3× the single-view count and can exceed max_model_len.
        # Halving the sampling_fps reduces frames/video from ~40 to ~20, keeping
        # 3 × ~20 × ~343 tokens/frame ≈ 20 580 well within max_model_len=32768.
        import attrs as _attrs  # noqa: PLC0415

        _mv_config = _attrs.evolve(self._window_config, sampling_fps=self._window_config.sampling_fps / 2.0)

        # Extract frames for every camera; make_windows_for_video attaches
        # Window objects to each clip in-place.
        per_camera_frames: list[list] = []
        for cam in self._MULTIVIEW_CAMERAS:
            video, _task = group[cam]
            _windows, frames = windowing_utils.make_windows_for_video(
                video,
                _mv_config,
                num_decode_threads,
                keep_mp4=self._keep_mp4,
            )
            per_camera_frames.append(frames)  # list of tensors, one per window

        # Build one multi-view model_input per window index on the FRONT clip.
        front_video, _front_task = group["CAM_FRONT"]
        num_windows = len(per_camera_frames[1])  # number of windows in FRONT video

        front_windows: list[Window] = []
        for clip in front_video.clips:
            front_windows.extend(clip.windows)

        for win_idx in range(num_windows):
            frames_list = [per_camera_frames[cam_idx][win_idx] for cam_idx in range(len(self._MULTIVIEW_CAMERAS))]
            llm_input = make_multiview_model_input(
                frames_list,
                self._vllm_config,
                self._processor,
                prompt,
            )
            if win_idx < len(front_windows):
                front_windows[win_idx].model_input[self._vllm_config.model_variant] = llm_input

    def _prep_windows(self, video: Video, prompt: str) -> None:
        """Prep the windows for the vLLM model.

        The videos are modified in-place by creating the windows
        for each clip in the videos and adding windows to each clip.

        Model inputs are added to each window.

        Args:
            video: The video to prep the windows for.
            prompt: The prompt to use for the vLLM model.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        num_video_decode_threads = max(1, int(self.resources.cpus) + 1)

        custom_windows = self._gt_provider.get_windows(video.input_path) if self._gt_provider else None

        windows, frames = windowing_utils.make_windows_for_video(
            video,
            self._window_config,
            num_video_decode_threads,
            keep_mp4=self._keep_mp4,
            custom_windows=custom_windows,
        )

        metadata = make_metadata(frames, self._window_config)

        # (Optional)Create debug identifiers for frame organization
        debug_window_ids = None
        if self._vllm_config.debug_save_frames:
            debug_window_ids = []
            for window in windows:
                # Find which clip this window belongs to
                clip_uuid = "unknown_clip"
                for clip in video.clips:
                    if window in clip.windows:
                        clip_uuid = str(clip.uuid)
                        break
                debug_window_ids.append(clip_uuid)
        
        #create the model inputs for the vLLM model
        llm_inputs = make_model_inputs(  
            frames,
            metadata,
            self._vllm_config,
            self._processor,
            prompt,
            debug_window_ids=debug_window_ids,
        )
        
        #stores the llm inputs 
        for window, llm_input in zip(windows, llm_inputs, strict=True):
            window.model_input[self._vllm_config.model_variant] = llm_input

    @nvtx.annotate("VllmPrepStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[T]) -> list[T]:
        """Prepare the data for the vLLM caption stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks. In multi-view mode, tasks are held in an internal
            buffer until a complete camera triplet is available; only complete
            triplets are returned. Incomplete groups are logged as warnings.

        """
        if self._processor is None:
            msg = "self._processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        prompt = get_prompt(
            self._vllm_config.prompt_variant,
            self._vllm_config.prompt_text,
            verbose=self._verbose,
        )

        if not self._multi_view:
            # --- Single-view path (unchanged behaviour) ---
            for task in tasks:
                major_size = task.get_major_size()
                self._timer.reinit(self, major_size)

                video = get_video_from_task(task)
                video.stage_timestamps["VllmPrepStage_start"] = time.time()

                with self._timer.time_process():
                    self._prep_windows(video, prompt)           #prepares the windows for the vLLM model, splits each clip into 256-frame windows

                video.stage_timestamps["VllmPrepStage_end"] = time.time()

                stage_perf = getattr(task, "stage_perf", None)
                if self._log_stats and stage_perf is not None:
                    stage_name, stage_perf_stats = self._timer.log_stats()
                    stage_perf[stage_name] = stage_perf_stats

            return tasks

        # --- Multi-view path ---
        # Store (Video, task) in the buffer keyed by scene_id and camera name.
        # Tasks must be stored here (not in a local dict) so they survive across
        # multiple process_data() calls when triplet partners arrive in later batches.
        for task in tasks:
            video = get_video_from_task(task)
            scene_id, camera = self._parse_scene_camera(video)
            self._scene_buffer.setdefault(scene_id, {})[camera] = (video, task)

        required_cameras = set(self._MULTIVIEW_CAMERAS)
        completed_tasks: list[T] = []

        for scene_id in list(self._scene_buffer.keys()):
            group = self._scene_buffer[scene_id]
            if not required_cameras.issubset(group.keys()):
                continue  # Wait for the remaining cameras to arrive in a future batch

            with self._timer.time_process():
                self._prep_multiview_windows(group, prompt)

            # Retrieve the stored tasks for all three cameras and return them.
            for cam in self._MULTIVIEW_CAMERAS:
                _video, stored_task = group[cam]
                completed_tasks.append(cast("T", stored_task))

            del self._scene_buffer[scene_id]

        # Incomplete groups remain in the buffer for future batches.
        if self._scene_buffer:
            waiting = {sid: list(cams.keys()) for sid, cams in self._scene_buffer.items()}
            logger.debug(f"Multi-view: waiting for partner cameras: {waiting}")

        return completed_tasks


class VllmCaptionStage(CuratorStage):
    """Stage that prepares video windows for vLLM multimodal model processing.

    This stage handles the preparation of video windows and prompts for vLLM-based models.
    """

    def __init__(  # noqa: PLR0913
        self,
        vllm_config: VllmConfig,
        max_inflight_requests: int = 0,
        *,
        inflight_batching: bool = False,
        keep_mp4: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the vLLM caption stage.

        Args:
            vllm_config: Configuration for the vLLM model.
            max_inflight_requests: Maximum number of inflight requests to vLLM
               engine. Set to 0 for unlimited inflight requests. Ignored if
               inflight_batching is False.
            inflight_batching: set to True to enable inflight batching.
            keep_mp4: Whether to keep the mp4 bytes.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        super().__init__()

        self._timer = StageTimer(self)
        self._vllm_config = vllm_config
        self._llm: LLM | None = None
        self._sampling_params: SamplingParams | None = None
        self._processor: AutoProcessor | None = None
        self._keep_mp4 = keep_mp4
        self._verbose = verbose
        self._log_stats = log_stats
        self._vllm_use_tqdm = False
        self._model = VllmModelInterface(self._vllm_config)
        self._max_inflight_requests = max_inflight_requests
        self._inflight_batching = inflight_batching

    def stage_setup_on_node(self) -> None:
        """Set up on a node by copying model weights if configured.

        This method copies model weights from the default cache location to a
        user-configured directory (e.g., local SSD) before loading the model.

        If the copy fails, the model will use the default cache location.

        """
        if self._vllm_config.copy_weights_to is None:
            logger.debug("No custom weights directory configured, skipping weight copy")
        else:
            for model_id in self._model.model_id_names:
                source_dir = model_utils.get_local_dir_for_weights_name(model_id)

                if not source_dir.exists():
                    msg = f"Source model weights directory does not exist: {source_dir}"
                    raise FileNotFoundError(msg)

                dest_dir = self._vllm_config.copy_weights_to / model_id
                logger.info(f"Copying model weights for {model_id} from {source_dir} to {dest_dir}")

                try:
                    model_utils.copy_model_weights(source_dir, dest_dir)
                    logger.info(f"Successfully copied model weights to {dest_dir}")
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to copy model weights, will use default location")

        # Instantiate vLLM Engine involves torch.compile which produces cache
        # To avoid conflict, do it once here
        gpu_stage_startup(f"{self.__class__.__name__}-on-node", self.resources.gpus, pre_setup=True)
        self._llm = vllm_model(self._vllm_config)
        gpu_stage_startup(f"{self.__class__.__name__}-on-node", self.resources.gpus, pre_setup=False)

    def stage_setup(self) -> None:
        """Set up the model for processing."""
        if self._llm is None:
            gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=True)
            self._llm = vllm_model(self._vllm_config)
        self._sampling_params = sampling_params(self._vllm_config.sampling_config)
        self._processor = auto_processor(self._vllm_config)
        gpu_stage_startup(self.__class__.__name__, self.resources.gpus, pre_setup=False)

    def destroy(self) -> None:
        """Clean up GPU resources."""
        gpu_stage_cleanup(self.__class__.__name__)

    def _reset(self) -> None:
        """Reset the vLLM model."""
        self._llm = None  # set to None (not del) so stage_setup can check `is None`
        self._sampling_params = None
        self._processor = None
        self.destroy()
        self.stage_setup()

    def secondary_name(self) -> str:
        """Get the secondary name of the stage.

        Returns:
            The secondary name of the stage.

        """
        # mypy is not smart enough to know that self._vllm_config.model_variant is a str
        # but mypy also thinks that this is a redundant cast
        return cast("str", self._vllm_config.model_variant)  # type: ignore[redundant-cast]

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name.

        Returns:
            The conda environment name.

        """
        return "unified"

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            The resource requirements for this stage.

        """
        return CuratorStageResource(gpus=self._vllm_config.num_gpus)

    @property
    def model(self) -> VllmModelInterface:
        """Get the model for this stage.

        Returns:
            The model for this stage.

        """
        return self._model

    @nvtx.annotate("VllmCaptionStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[T]) -> list[T]:  # noqa: C901           
        """Process the data for the vLLM caption stage.

        Args:
            tasks: The tasks to process.

        Returns:
            The processed tasks.

        """
        if self._llm is None:
            msg = "vLLM model not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._sampling_params is None:
            msg = "Sampling parameters not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        if self._processor is None:
            msg = "Processor not initialized, call stage_setup() first"
            raise RuntimeError(msg)

        major_size = sum(task.get_major_size() for task in tasks)
        self._timer.reinit(self, major_size)

        @tenacity.retry(stop=tenacity.stop_after_attempt(self._vllm_config.max_retries), reraise=True)
        def _vllm_caption(
            model_inputs: list[dict[str, Any]], stage2_prompts: list[str | None]
        ) -> tuple[list[str], list[str | None]]:
            try:
                assert self._processor is not None
                caption_result = vllm_caption(
                    model_inputs,
                    self._llm,
                    self._processor,
                    self._sampling_params,
                    self._vllm_config,
                    inflight_batching=self._inflight_batching,
                    max_inflight_requests=self._max_inflight_requests,
                    stage2_prompts=stage2_prompts,
                )
            except Exception as e:
                input_videos = [str(get_video_from_task(t).input_video) for t in tasks]
                input_videos_str = ", ".join(input_videos)
                logger.exception(f"Error generating captions for video {input_videos_str}, trying again")
                for task in tasks:
                    video = get_video_from_task(task)
                    video.errors["captioning"] = f"vLLM captioning error: {e}"

                # On retry: tear down and restart vllm
                self._reset()
                raise
            else:
                for task in tasks:
                    video = get_video_from_task(task)
                    video.errors.pop("captioning", None)

                return caption_result

        _caption_stage_start = time.time()
        with self._timer.time_process():            
            # Gather model inputs and clip uuids
            windows, clip_uuids = _get_windows_from_tasks(tasks)     

            # Filter to only windows that have model_input for this variant.
            # In multi-view mode, only FRONT clip windows have model_input populated;
            # FRONT_LEFT and FRONT_RIGHT windows pass through without captioning.
            variant = self._vllm_config.model_variant
            active_pairs = [(w, u) for w, u in zip(windows, clip_uuids) if variant in w.model_input]
            active_windows = [p[0] for p in active_pairs]
            active_clip_uuids = [p[1] for p in active_pairs]
            model_inputs = [w.model_input[variant] for w in active_windows]

            # Set up stage 2 prompts if enabled
            stage2_prompts = _get_stage2_prompts(self._vllm_config, len(active_windows))

            # Generate captions (and reasoning traces for reasoning models)
            try:
                captions, reasonings = (
                    _vllm_caption(model_inputs, stage2_prompts) if active_windows else ([], [])
                )
            except Exception:  # noqa: BLE001
                logger.error(f"All {self._vllm_config.max_retries} retry attempts exhausted, returning empty captions")
                captions = [""] * len(model_inputs)
                reasonings = [None] * len(model_inputs)

            # Scatter captions back to windows
            _scatter_captions(active_windows, captions, active_clip_uuids, variant, verbose=self._verbose)
            _scatter_reasoning(active_windows, reasonings, variant)

            logger.info(f"Generated {len(captions)} captions for {len(tasks)} tasks")

        _caption_stage_end = time.time()
        for task in tasks:
            _v = get_video_from_task(task)
            _v.stage_timestamps["VllmCaptionStage_start"] = _caption_stage_start
            _v.stage_timestamps["VllmCaptionStage_end"] = _caption_stage_end

        if self._log_stats:
            # Because there's a single call to caption all tasks, just log the first task's stage_perf.
            stage_name, stage_perf_stats = self._timer.log_stats()
            stage_perf = getattr(tasks[0], "stage_perf", None)
            if stage_perf is not None:
                stage_perf[stage_name] = stage_perf_stats

        _free_vllm_inputs(windows, self._vllm_config.model_variant, keep_mp4=self._keep_mp4)
        return tasks
