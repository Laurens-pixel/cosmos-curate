# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Marlin-2B dense-caption pipeline stage.

NemoStation/Marlin-2B produces timestamped events for a whole clip via
``model.caption(mp4_path)``.  This stage maps those events onto the pipeline's
existing window boundaries (256-frame fixed-stride or GT windows) by finding
the majority-overlap event description for each window.

sys.path injection order (same as test_marlin.py):
  1. unified env site-packages  (torch, tokenizers, qwen-vl-utils, av, …)
  2. pip_overrides              (transformers 5.5.0 with Qwen3_5 support)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any

from loguru import logger

MODEL_DIR = "/config/models/NemoStation/Marlin-2B"
_UNIFIED_SITE = "/opt/cosmos-curate/.pixi/envs/unified/lib/python3.12/site-packages"
_PIP_OVERRIDES = "/config/pip_overrides"
_CAPTION_KEY = "marlin"


def _inject_sys_path() -> None:
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "av")
    for p in (_PIP_OVERRIDES, _UNIFIED_SITE):
        if p in sys.path:
            sys.path.remove(p)
    sys.path = [_PIP_OVERRIDES, _UNIFIED_SITE] + sys.path


def _event_overlap(event: dict[str, Any], win_start_s: float, win_end_s: float) -> float:
    return max(0.0, min(event["end"], win_end_s) - max(event["start"], win_start_s))


# ── Stage ─────────────────────────────────────────────────────────────────────

from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curate.pipelines.video.captioning.gt_window_provider import GTWindowProvider, make_gt_window_provider
from cosmos_curate.pipelines.video.utils import windowing_utils
from cosmos_curate.pipelines.video.utils.data_model import Window, WindowConfig


class MarlinCaptionStage(CuratorStage):
    """Run Marlin-2B on each clip and assign event descriptions to windows."""

    def __init__(
        self,
        window_config: WindowConfig,
        model_dir: str = MODEL_DIR,
        keep_mp4: bool = False,
        gt_window_source: str | None = None,
        gt_window_cfg: dict | None = None,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        self._window_config = window_config
        self._model_dir = model_dir
        self._keep_mp4 = keep_mp4
        self._gt_window_source = gt_window_source
        self._gt_window_cfg = gt_window_cfg or {}
        self._verbose = verbose
        self._log_stats = log_stats
        self._model = None
        self._processor = None
        self._gt_provider: GTWindowProvider | None = None

    @property
    def resources(self) -> CuratorStageResource:
        return CuratorStageResource(gpus=1.0, cpus=4)

    @property
    def conda_env_name(self) -> str:
        return "unified"

    def stage_setup(self) -> None:
        _inject_sys_path()

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"MarlinCaptionStage: loading model from {self._model_dir}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=None,
            local_files_only=True,
        )
        self._model = self._model.to("cuda").eval()
        self._processor = AutoProcessor.from_pretrained(
            self._model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model._processor = self._processor
        if self._gt_window_source is not None:
            self._gt_provider = make_gt_window_provider(self._gt_window_source, self._gt_window_cfg)
        logger.info("MarlinCaptionStage: model ready")

    def process_data(self, tasks: list[Any]) -> list[Any]:
        for task in tasks:
            video = task.video
            t0 = time.monotonic()

            # Determine GT windows if a provider is configured.
            custom_windows = self._gt_provider.get_windows(video.input_path) if self._gt_provider else None

            if custom_windows is not None:
                # GT boundaries available: create windows aligned to action segments.
                windowing_utils.make_windows_for_video(
                    video,
                    self._window_config,
                    num_decode_threads=4,
                    keep_mp4=self._keep_mp4,
                    return_frames=False,
                    custom_windows=custom_windows,
                )

            for clip in video.clips:
                data = clip.encoded_data.resolve()
                if data is None or data.nbytes == 0:
                    logger.warning("MarlinCaptionStage: empty encoded_data for clip")
                    continue

                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/tmp", delete=False) as f:
                        f.write(bytes(data))
                        tmp_path = f.name
                    result = self._model.caption(tmp_path)
                except Exception as exc:
                    logger.error(f"MarlinCaptionStage: caption() failed: {exc}")
                    result = {}
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                events: list[dict[str, Any]] = result.get("events") or []
                scene: str = result.get("scene", "")

                if custom_windows is None:
                    # No GT boundaries: treat the whole clip as one window and join
                    # all Marlin events into a single caption.
                    caption = " ".join(ev.get("description", "") for ev in events).strip() or scene
                    if caption:
                        meta = clip.extract_metadata()
                        total_frames = int(meta.get("num_frames") or 1) if meta else 1
                        window = Window(start_frame=0, end_frame=max(total_frames - 1, 0))
                        if self._keep_mp4:
                            window.mp4_bytes = bytes(data)
                        clip.windows.append(window)
                        window.caption[_CAPTION_KEY] = caption
                else:
                    meta = clip.extract_metadata()
                    fps: float = meta["framerate"] if meta and meta.get("framerate") else 30.0
                    for window in clip.windows:
                        win_start_s = window.start_frame / fps
                        win_end_s = window.end_frame / fps
                        overlapping = sorted(
                            [ev for ev in events if _event_overlap(ev, win_start_s, win_end_s) > 0.0],
                            key=lambda e: e["start"],
                        )
                        caption = " ".join(ev.get("description", "") for ev in overlapping).strip() or scene
                        if caption:
                            window.caption[_CAPTION_KEY] = caption

            if self._log_stats:
                logger.info(f"MarlinCaptionStage: {video.input_path} done in {time.monotonic()-t0:.1f}s")

        return tasks
