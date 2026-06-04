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
"""C-RADIOv4-H image embedding model for video clip embeddings.

Processes video frames independently through C-RADIOv4-H's ViT-H backbone,
using cpe_video_mode for consistent positional encoding across frames,
then mean-pools per-frame summary embeddings into a single clip embedding.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from cosmos_curate.core.interfaces.model_interface import ModelInterface
from cosmos_curate.core.utils.model import model_utils

_CRADIO_MODEL_ID: Final = "nvidia/C-RADIOv4-H"
_TARGET_NUM_FRAMES: Final = 8
_TARGET_RESOLUTION: Final = 432  # Multiple of 16, good balance of quality vs memory


class CRadio(ModelInterface):
    """C-RADIOv4-H embedding model for video clip processing."""

    def __init__(self, *, utils_only: bool = False, target_resolution: int = _TARGET_RESOLUTION) -> None:
        """Initialize the C-RADIOv4-H model.

        Args:
            utils_only: If True, skip loading the model (for CPU-only frame preparation).
            target_resolution: Target resolution for frame resizing (must be multiple of 16).

        """
        super().__init__()
        self._utils_only = utils_only
        self._target_resolution = target_resolution
        self._model: torch.nn.Module | None = None

    @property
    def conda_env_name(self) -> str:
        """Get the conda environment name."""
        return "unified"

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names."""
        return [_CRADIO_MODEL_ID]

    def setup(self) -> None:
        """Set up the C-RADIOv4-H model."""
        self._weights_dir = str(model_utils.get_local_dir_for_weights_name(_CRADIO_MODEL_ID))
        if not Path(self._weights_dir).exists():
            msg = f"Weights directory {self._weights_dir} not found!"
            raise FileNotFoundError(msg)

        logger.info(f"Setting up C-RADIOv4-H model from {self._weights_dir}")

        if not self._utils_only:
            from transformers import AutoModel

            # Pre-populate HF cache with model's custom code to avoid
            # race conditions when multiple workers load simultaneously.
            self._prepopulate_hf_module_cache()

            self._model = AutoModel.from_pretrained(
                self._weights_dir,
                trust_remote_code=True,
                local_files_only=True,
            ).to("cuda")
            self._model.eval()
        else:
            self._model = None

    def _prepopulate_hf_module_cache(self) -> None:
        """Copy all .py files from model dir to HF transformers module cache.

        With trust_remote_code=True, transformers copies .py files from the
        model directory to its cache and loads from there. When multiple workers
        do this concurrently, some files can be missed. Pre-copying prevents this.
        """
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            msg = "HF_HOME environment variable must be set to avoid writing to home directory"
            raise RuntimeError(msg)
        dir_name = Path(self._weights_dir).name
        safe_name = dir_name.replace("-", "_hyphen_")
        cache_dir = Path(hf_home) / "modules" / "transformers_modules" / safe_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        for py_file in Path(self._weights_dir).glob("*.py"):
            dest = cache_dir / py_file.name
            if not dest.exists():
                shutil.copy2(py_file, dest)
                logger.debug(f"Pre-copied {py_file.name} to HF module cache")

    def get_target_num_frames(self) -> int:
        """Get the target number of frames for the model."""
        return _TARGET_NUM_FRAMES

    def formulate_input_frames(self, frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32] | None:
        """Prepare input frames for the model.

        Samples target number of frames, resizes to target resolution,
        and converts to float32 tensor in [0, 1] range with shape (T, C, H, W).

        Args:
            frames: List of video frames as uint8 arrays (H, W, C).

        Returns:
            Preprocessed frames as float32 array (T, C, H, W) in [0, 1] range,
            or None if insufficient frames.

        """
        fn = self.get_target_num_frames()
        if len(frames) < fn:
            logger.error(f"Frame count {len(frames)} is smaller than minimal requirement {fn}")
            return None

        # Subsample to target number of frames
        step = len(frames) // fn
        sampled = frames[::step][:fn]

        # Resize and normalize
        res = self._target_resolution
        processed = []
        for frame in sampled:
            resized = cv2.resize(frame, (res, res))  # type: ignore[misc]
            # Convert HWC uint8 -> CHW float32 in [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
            processed.append(transposed)

        return np.stack(processed, axis=0)  # (T, C, H, W)

    def encode_video_frames(self, frames: npt.NDArray[np.float32]) -> torch.Tensor:
        """Encode video frames to produce a single clip embedding.

        Uses cpe_video_mode for consistent positional encoding across frames,
        then mean-pools per-frame summary embeddings.

        Args:
            frames: Preprocessed frames as float32 array (T, C, H, W) in [0, 1] range.

        Returns:
            Mean-pooled, L2-normalized clip embedding tensor (1, D).

        """
        assert self._model is not None

        if frames.size == 0:
            return torch.empty((1, 0), dtype=torch.float16)

        num_frames = frames.shape[0]
        frames_tensor = torch.from_numpy(frames).to("cuda", dtype=torch.float32)

        with torch.no_grad():
            # Use cpe_video_mode for consistent position encoding across frames
            with self._model.radio_model.cpe_video_mode(t=num_frames):
                summary, _features = self._model(frames_tensor)

            # summary shape: (T, D) - one summary per frame
            # Mean-pool across temporal dimension
            clip_embedding = summary.mean(dim=0, keepdim=True)  # (1, D)

            # L2 normalize
            clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)

        return clip_embedding.to("cpu", dtype=torch.float16)

    def encode_batched_videos(
        self,
        videos: list[npt.NDArray[np.float32]],
        batch_size: int = 4,
    ) -> list[npt.NDArray[np.float32]]:
        """Encode a batch of videos.

        Args:
            videos: List of preprocessed frame arrays, each (T, C, H, W).
            batch_size: Number of videos to process at once.

        Returns:
            List of per-video embeddings as numpy arrays.

        """
        embeddings: list[npt.NDArray[np.float32]] = []
        for i in range(0, len(videos), batch_size):
            batch = videos[i : i + batch_size]
            for video_frames in batch:
                embedding = self.encode_video_frames(video_frames)
                embeddings.append(embedding.numpy())
        return embeddings
