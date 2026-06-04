# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 video-judge plugins (E4B and 31B).

Extends the text-only Gemma 4 judges in judge_gemma4.py with video evidence:
the model receives decoded frames alongside the caption, enabling detection of
object/action errors that require visual confirmation.

Evidence kind: ``mp4_bytes`` — frames are decoded inside ``judge_batch`` via PyAV.
Output format: CORRECT / INCORRECT (same as the text judge, same verdict parser).

Reference performance of the text-only variants (30 videos, 125 windows):
  - gemma4_e4b  (text):  F1=0.655, P=65.5%, R=65.5%
  - gemma4_31b  (text):  not yet evaluated on this benchmark
"""

from typing import Any

import torch
from loguru import logger
from PIL import Image

from cosmos_curate.core.interfaces.stage_interface import CuratorStageResource
from cosmos_curate.models.judge_gemma4 import JudgeGemma4Base, _parse_verdict, _verdict_to_score
from cosmos_curate.models.judge_plugin import EvidenceKind, JudgeItem, JudgeResult
from cosmos_curate.models.judge_video_utils import decode_frames

# ── Frame-sampling parameters ─────────────────────────────────────────────────
# 4 fps × 8.5 s window ≈ 34 frames — stays above Gemma4's 32-frame minimum.
# 448 px matches the resolution used by Gemma4DirectCaptionStage.

_SAMPLING_FPS: float = 4.0
_MAX_FRAMES: int = 32
_FRAME_SIZE: int = 448

# ── Prompt templates ──────────────────────────────────────────────────────────
# All templates use {caption_text}; GT variants also use {gt_action_text}.
# Extra kwargs are silently ignored by str.format(), so it is always safe to
# call template.format(caption_text=..., gt_action_text=...).

_GEMMA4_VIDEO_PROMPTS: dict[str, str] = {
    # --- Robot manipulation (with GT) ---
    "lenient_binary": (
        "Watch the video carefully and identify what the robotic arm is doing.\n\n"
        "<caption>{caption_text}</caption>\n\n"
        "Ground truth action: {gt_action_text}\n\n"
        "This video shows a robotic arm performing a manipulation task. "
        "Verify whether the caption correctly describes the action shown.\n\n"
        "Judge as CORRECT if:\n"
        "- The caption identifies the same object, even loosely "
        "(\"purple vegetable\" for \"onion\", \"bottle\" for \"green tea bottle\" — all fine)\n"
        "- The caption conveys the same overall action goal as the ground truth\n\n"
        "Judge as INCORRECT only if:\n"
        "- The caption clearly names a different object\n"
        "- The caption describes the opposite action (placing when the GT is retrieving)\n\n"
        "Respond with exactly one word on the first line — CORRECT or INCORRECT — "
        "then one sentence describing what you actually saw in the video."
    ),
    # --- Robot manipulation (reference-free) ---
    "lenient_binary_no_gt": (
        "Watch the video carefully and identify what the robotic arm is doing.\n\n"
        "<caption>{caption_text}</caption>\n\n"
        "This video shows a robotic arm performing a manipulation task. "
        "Based only on what you see in the video, check whether the caption is accurate.\n\n"
        "Judge as CORRECT if the caption correctly identifies the object being manipulated "
        "and the overall action goal (retrieve, place, push, grasp). "
        "Approximate or vague object descriptions are acceptable.\n\n"
        "Judge as INCORRECT if the caption names a clearly different object or describes "
        "the opposite action from what you see.\n\n"
        "Respond with exactly one word on the first line — CORRECT or INCORRECT — "
        "then one sentence describing what you actually saw in the video."
    ),
    # --- Cooking / YouCook2 (with GT) ---
    "youcook2_binary": (
        "Watch the video carefully and identify the cooking action being performed.\n\n"
        "<caption>{caption_text}</caption>\n\n"
        "Ground truth action: {gt_action_text}\n\n"
        "This video shows a cooking step from a recipe. "
        "Verify whether the caption correctly identifies the cooking action.\n\n"
        "Judge as CORRECT if:\n"
        "- The caption describes the same cooking action as the ground truth "
        "(\"slicing\" for \"cutting\" is fine)\n"
        "- Ingredient differences or omissions are acceptable — the action is what matters\n\n"
        "Judge as INCORRECT only if the caption describes a completely different cooking action.\n\n"
        "Respond with exactly one word on the first line — CORRECT or INCORRECT — "
        "then one sentence describing what you actually saw in the video."
    ),
    # --- Cooking / YouCook2 (reference-free) ---
    "youcook2_binary_no_gt": (
        "Watch the video carefully and identify the cooking action being performed.\n\n"
        "<caption>{caption_text}</caption>\n\n"
        "This video shows a cooking step from a recipe. "
        "Based only on what you see in the video, check whether the caption accurately "
        "describes the cooking action.\n\n"
        "Judge as CORRECT if the caption identifies the right cooking action "
        "(chopping, boiling, frying, mixing, etc.). "
        "Ingredient specifics are secondary — approximate descriptions are fine.\n\n"
        "Judge as INCORRECT if the caption describes a completely different cooking action.\n\n"
        "Respond with exactly one word on the first line — CORRECT or INCORRECT — "
        "then one sentence describing what you actually saw in the video."
    ),
}

_DEFAULT_PROMPT = _GEMMA4_VIDEO_PROMPTS["lenient_binary"]


# ── Base class ────────────────────────────────────────────────────────────────


class JudgeGemma4VideoBase(JudgeGemma4Base):
    """Gemma 4 judge with video evidence (mp4_bytes → PIL frames → multimodal inference).

    Subclasses override ``variant()`` and ``resources`` / ``_load_kwargs`` for
    E4B vs 31B.  Model loading is fully inherited from ``JudgeGemma4Base.setup()``.
    """

    # ── Capability override ───────────────────────────────────────────────────

    @property
    def evidence_kind(self) -> EvidenceKind:
        """Video judge needs the window MP4 bytes to decode frames."""
        return "mp4_bytes"

    # ── Inference ─────────────────────────────────────────────────────────────

    def _build_video_text(self, frames: list[Image.Image], caption: str, gt_action_text: str = "") -> str:
        """Build formatted prompt text for one item (frames required for correct token count)."""
        assert self._processor is not None
        assert self._config is not None
        variant_key = self._config.prompt_variant
        if not gt_action_text:
            no_gt_key = variant_key + "_no_gt"
            if no_gt_key in _GEMMA4_VIDEO_PROMPTS:
                variant_key = no_gt_key
        template = _GEMMA4_VIDEO_PROMPTS.get(variant_key, _DEFAULT_PROMPT)
        prompt = template.format(caption_text=caption, gt_action_text=gt_action_text)
        messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]}]
        return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def judge_batch(self, items: list[JudgeItem]) -> list[JudgeResult]:
        """True batched GPU inference: one processor call + one model.generate() for all items."""
        assert self._model is not None
        assert self._processor is not None
        assert self._input_device is not None
        assert self._config is not None

        results: list[JudgeResult | None] = [None] * len(items)
        valid_indices: list[int] = []
        valid_frames: list[list[Image.Image]] = []
        valid_texts: list[str] = []

        for i, item in enumerate(items):
            if item.mp4_bytes is None:
                results[i] = JudgeResult(verdict=None, score=1, explanation="no mp4_bytes")
                continue
            try:
                frames = decode_frames(item.mp4_bytes, sampling_fps=_SAMPLING_FPS, max_frames=_MAX_FRAMES, frame_size=_FRAME_SIZE)
                if not frames:
                    results[i] = JudgeResult(verdict=None, score=1, explanation="0 frames decoded")
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"JudgeGemma4Video[{self.variant()}] frame decode failed: {exc}")
                results[i] = JudgeResult(verdict=None, score=1, explanation=str(exc))
                continue
            valid_indices.append(i)
            valid_frames.append(frames)
            valid_texts.append(self._build_video_text(frames, item.caption, item.gt_action_text or ""))

        if not valid_indices:
            return [r or JudgeResult(verdict=None, score=1, explanation="skipped") for r in results]

        try:
            # Pad shorter clips to uniform length by repeating the last frame
            n_frames = max(len(f) for f in valid_frames)
            padded_frames = [f + [f[-1]] * (n_frames - len(f)) for f in valid_frames]

            inputs = self._processor(
                text=valid_texts,
                videos=padded_frames,
                return_tensors="pt",
                padding=True,
                num_frames=n_frames,  # override Gemma4VideoProcessor class-level default of 32
            ).to(self._input_device)

            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)

            for orig_i, raw in zip(valid_indices, decoded):
                verdict, explanation = _parse_verdict(raw)
                results[orig_i] = JudgeResult(
                    verdict=verdict,
                    score=_verdict_to_score(verdict),
                    explanation=explanation,
                    raw_output=raw,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"JudgeGemma4Video[{self.variant()}] batch inference failed: {exc}")
            for orig_i in valid_indices:
                if results[orig_i] is None:
                    results[orig_i] = JudgeResult(verdict=None, score=1, explanation=str(exc))

        return [r or JudgeResult(verdict=None, score=1, explanation="unknown error") for r in results]


# ── Concrete plugins ──────────────────────────────────────────────────────────


class JudgeGemma4E4BVideo(JudgeGemma4VideoBase):
    """Gemma 4 E4B-it with video evidence. ~9 GB VRAM + frame buffers → 1 full GPU."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "gemma4_e4b_video"

    @property
    def resources(self) -> CuratorStageResource:
        """1 GPU — video frame processing needs more headroom than text-only (0.5 GPU)."""
        return CuratorStageResource(cpus=4.0, gpus=1.0)


class JudgeGemma4_31BVideo(JudgeGemma4VideoBase):
    """Gemma 4 31B-it with video evidence. Sharded across 2 A100s via device_map='auto'."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "gemma4_31b_video"

    @property
    def resources(self) -> CuratorStageResource:
        """2 GPUs — 62 GB weights sharded across both A100s."""
        return CuratorStageResource(cpus=2.0, gpus=2.0)

    def _load_kwargs(self) -> dict[str, Any]:
        """Use ``device_map='auto'`` so accelerate places layers across both GPUs."""
        return {"device_map": "auto"}
