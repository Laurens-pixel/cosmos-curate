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
"""Direct HuggingFace-based Gemma4 captioning stage.

Replaces VllmPrepStage + VllmCaptionStage for the gemma4 algorithm.
Uses AutoModelForImageTextToText directly — no vLLM required.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import WindowConfig

if TYPE_CHECKING:
    from cosmos_curate.pipelines.video.utils.data_model import SplitPipeTask

MODEL_VARIANT = "gemma4"
NUM_FRAMES = 16          # frames sampled per window (kept for reference; actual count from sampling_fps)
MAX_NEW_TOKENS = 512
SAMPLING_FPS_DEFAULT = 4.0   # 4fps × 8.5s window ≈ 34 frames; covers full pick-place arc


class Gemma4DirectCaptionStage(CuratorStage):
    """Caption stage using HuggingFace Gemma4 directly (no vLLM).

    Performs window splitting, frame extraction, and caption generation
    in a single stage, replacing VllmPrepStage + VllmCaptionStage.
    """

    def __init__(
        self,
        window_config: WindowConfig,
        prompt_variant: str = "default",
        max_new_tokens: int = MAX_NEW_TOKENS,
        sampling_fps: float = SAMPLING_FPS_DEFAULT,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialise the stage."""
        import attrs
        # Override sampling_fps so make_windows_for_video extracts more frames per window
        self._window_config = attrs.evolve(window_config, sampling_fps=sampling_fps)
        self._prompt_variant = prompt_variant
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose
        self._log_stats = log_stats
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device | None = None

    @property
    def resources(self) -> CuratorStageResource:
        """Request 1 GPU per worker — model is ~9 GB in bfloat16."""
        return CuratorStageResource(gpus=1.0)

    @property
    def pixi_environment(self) -> str:
        """Use the unified env which has torch + transformers."""
        return "unified"

    def _model_path(self) -> Path:
        """Return the local model path."""
        hf_home = Path("/config/models")
        return hf_home / "google" / "gemma-4-E4B-it"

    def stage_setup(self) -> None:
        """Load Gemma4 model and processor."""
        import os
        import sys

        # The Ray worker for the unified env may not have all unified env packages
        # in sys.path at the point when stage_setup() runs.  Explicitly add the
        # unified site-packages so that the `tokenizers` C-extension is importable
        # before we inject pip_overrides.  Without tokenizers, pip_overrides'
        # transformers/__init__.py would cache is_tokenizers_available()=False and
        # create a _DummyObject for GemmaTokenizerFast.
        unified_sp = "/opt/cosmos-curate/.pixi/envs/unified/lib/python3.12/site-packages"
        if unified_sp not in sys.path:
            sys.path.insert(0, unified_sp)

        # Inject pip_overrides AFTER unified_sp so transformers from pip_overrides
        # (5.5.0, which has Gemma4 support) takes priority over unified env's 4.57.6,
        # while still finding tokenizers from unified_sp.
        pip_overrides = "/config/pip_overrides"
        if pip_overrides not in sys.path:
            sys.path.insert(0, pip_overrides)

        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
        from transformers.models.gemma4.processing_gemma4 import Gemma4Processor

        # windowing_utils.py conditionally imports fetch_video only when
        # CONDA_DEFAULT_ENV == "unified", which isn't set in the Ray worker.
        # Patch the module namespace so the function is available at call time.
        from cosmos_curate.pipelines.video.utils import windowing_utils as _wutils
        if not hasattr(_wutils, "fetch_video"):
            from cosmos_curate.pipelines.video.utils.vision_process import fetch_video as _fv
            _wutils.fetch_video = _fv

        model_path = str(self._model_path())
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Gemma4DirectCaptionStage] Loading processor from {model_path}")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._processor = Gemma4Processor.from_pretrained(
            model_path,
            local_files_only=True,
        )

        logger.info(f"[Gemma4DirectCaptionStage] Loading model from {model_path}")
        self._model = Gemma4ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(self._device)
        self._model.eval()
        logger.info("[Gemma4DirectCaptionStage] Model loaded successfully.")

    def destroy(self) -> None:
        """Free GPU memory."""
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_prompt(self) -> str:
        """Return the captioning prompt."""
        from cosmos_curate.models.prompts import get_prompt
        return get_prompt(self._prompt_variant, prompt_text=None)

    def _caption_window_frames(self, frames: torch.Tensor) -> str:
        """Run Gemma4 inference on a window's frames and return the caption string.

        Args:
            frames: Float tensor of shape (T, C, H, W), values in [0, 255].

        Returns:
            Generated caption text.

        """
        prompt_text = self._get_prompt()

        # Build the chat message — {"type": "video"} is the Gemma4 video placeholder
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Apply chat template to get text input
        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        # frames tensor: (T, C, H, W) float — processor expects list of PIL images
        import numpy as np
        from PIL import Image
        frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C), values in [0, 255]
        logger.info(
            f"[Gemma4] frames shape={frames_np.shape} dtype={frames_np.dtype} "
            f"min={frames_np.min():.1f} max={frames_np.max():.1f} mean={frames_np.mean():.1f}"
        )
        frames_np = frames_np.clip(0, 255).astype(np.uint8)
        frames_list = [Image.fromarray(f) for f in frames_np]  # list of PIL RGB images

        inputs = self._processor(
            text=text,
            videos=frames_list,
            return_tensors="pt",
            num_frames=len(frames_list),  # override class-level default of 32
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Split clips into windows, generate captions, store on each window.

        Args:
            tasks: Pipeline tasks containing video clips.

        Returns:
            Tasks with captions stored on each clip's windows.

        """
        if self._model is None or self._processor is None:
            msg = "stage_setup() must be called before process_data()"
            raise RuntimeError(msg)

        num_decode_threads = 4

        for task in tasks:
            video = task.video
            t_start = time.time()

            windows, frames = windowing_utils.make_windows_for_video(
                video,
                self._window_config,
                num_decode_threads,
                keep_mp4=False,
                return_frames=True,
            )

            if not windows:
                logger.warning(f"[Gemma4DirectCaptionStage] No windows for video {video.input_video}")
                continue

            for window, frame_tensor in zip(windows, frames):
                if frame_tensor is None or frame_tensor.numel() == 0:
                    logger.warning(
                        f"[Gemma4DirectCaptionStage] Empty frame tensor for window "
                        f"{window.start_frame}-{window.end_frame}"
                    )
                    window.errors["frames"] = "empty"
                    continue

                try:
                    caption = self._caption_window_frames(frame_tensor)
                    window.caption[MODEL_VARIANT] = caption
                    if self._verbose:
                        logger.info(
                            f"[Gemma4DirectCaptionStage] window {window.start_frame}-{window.end_frame}: "
                            f"{caption[:80]}..."
                        )
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"[Gemma4DirectCaptionStage] Caption failed for window "
                        f"{window.start_frame}-{window.end_frame}: {e}"
                    )
                    window.errors["caption"] = str(e)

            elapsed = time.time() - t_start
            logger.info(
                f"[Gemma4DirectCaptionStage] {video.input_video}: "
                f"{len(windows)} windows captioned in {elapsed:.1f}s"
            )

        return tasks
