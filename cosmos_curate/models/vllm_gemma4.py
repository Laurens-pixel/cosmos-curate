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
"""Gemma 4 vLLM plugin.

google/gemma-4-E4B-it is a 4.5B active / 8B total MoE multimodal model
(Apache 2.0).  Input format mirrors Qwen: a single {"type": "video"} marker
in the chat template produces the video_token_id placeholder (258884), which
vLLM expands to the per-frame soft tokens (280 tokens/frame) at inference.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoProcessor
from vllm import LLM

from cosmos_curate.models.vllm_plugin import VllmPlugin
from cosmos_curate.pipelines.video.utils.data_model import VllmCaptionRequest

if TYPE_CHECKING:
    import torch
    from vllm import RequestOutput

    from cosmos_curate.pipelines.video.utils.data_model import VllmConfig

MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85
MAX_NUM_BATCHED_TOKENS = 4096
TRUST_REMOTE_CODE = False
LIMIT_MM_PER_PROMPT = {"video": 1}

_DEFAULT_REFINE_PROMPT = """
Improve and refine following video description. Focus on highlighting the key visual and sensory elements.
Ensure the description is clear, precise, and paints a compelling picture of the scene.
"""


def make_message(text_input: str) -> dict[str, Any]:
    """Create a user message for Gemma 4 with a single video placeholder."""
    return {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": text_input},
        ],
    }


def make_prompt(
    message: dict[str, Any],
    frames: torch.Tensor,
    processor: AutoProcessor,
) -> dict[str, Any]:
    """Tokenize the chat message and return vLLM-ready input dict.

    The chat template inserts a video_token_id placeholder; vLLM replaces it
    with the actual per-frame visual embeddings at inference time.
    """
    prompt_ids = processor.apply_chat_template(  # type: ignore[attr-defined]
        [message],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()

    return {
        "prompt_token_ids": prompt_ids,
        "multi_modal_data": {"video": frames},
    }


class VllmGemma4(VllmPlugin):
    """Gemma 4 E4B-it vLLM plugin."""

    @staticmethod
    def model_variant() -> str:
        """Return the model variant name."""
        return "gemma4"

    @classmethod
    def processor(cls, config: VllmConfig) -> AutoProcessor:
        """Return the AutoProcessor for Gemma 4."""
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            cls.model_path(config),
            trust_remote_code=TRUST_REMOTE_CODE,
            use_fast=True,
        )
        return cast("AutoProcessor", processor)

    @classmethod
    def model(cls, config: VllmConfig) -> LLM:
        """Instantiate the vLLM model for Gemma 4.

        FP8 quantization is intentionally skipped — Gemma 4's MoE architecture
        uses mixed dtypes that are not compatible with FP8 in vLLM 0.11.x.
        """
        return LLM(
            model=str(cls.model_path(config)),
            limit_mm_per_prompt=LIMIT_MM_PER_PROMPT,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            tensor_parallel_size=config.num_gpus,
            trust_remote_code=TRUST_REMOTE_CODE,
            mm_processor_cache_gb=0.0 if config.disable_mmcache else 4.0,
            enable_chunked_prefill=True,
            enforce_eager=True,
        )

    @staticmethod
    def make_llm_input(
        prompt: str,
        frames: torch.Tensor,
        metadata: dict[str, Any],  # noqa: ARG004
        processor: AutoProcessor,
    ) -> dict[str, Any]:
        """Build vLLM input for a single video window."""
        message = make_message(prompt)
        return make_prompt(message, frames, processor)

    @staticmethod
    def make_refined_llm_input(
        caption: str,
        prev_input: dict[str, Any],
        processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build a stage-2 refinement input reusing the original video frames."""
        _refine_prompt = _DEFAULT_REFINE_PROMPT if refine_prompt is None else refine_prompt
        final_prompt = _refine_prompt + caption

        if "multi_modal_data" not in prev_input or "video" not in prev_input["multi_modal_data"]:
            msg = "prev_input does not contain multi_modal_data.video"
            raise ValueError(msg)

        video_frames = prev_input["multi_modal_data"]["video"]
        message = make_message(final_prompt)
        return make_prompt(message, video_frames, processor)

    @staticmethod
    def make_refined_llm_request(
        request: VllmCaptionRequest,
        processor: AutoProcessor,
        refine_prompt: str | None = None,
    ) -> VllmCaptionRequest:
        """Create a stage-2 VllmCaptionRequest from a completed stage-1 request."""
        if request.caption is None:
            msg = "Request caption is None"
            raise ValueError(msg)

        inputs = VllmGemma4.make_refined_llm_input(
            request.caption, request.inputs, processor, refine_prompt
        )
        return VllmCaptionRequest(
            request_id=secrets.token_hex(8),
            inputs=inputs,
        )

    @staticmethod
    def decode(vllm_output: RequestOutput) -> str:
        """Decode vLLM output into a caption string."""
        return vllm_output.outputs[0].text
