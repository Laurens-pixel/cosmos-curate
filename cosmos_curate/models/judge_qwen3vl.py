# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL judge plugin (30B-A3B Instruct, MoE).

A stronger, more transparent replacement for the VCInspector LoRA judge:

* No fine-tune — uses the off-the-shelf instruction model, so it inherits Qwen3-VL's
  general instruction-following rather than ActivityNet-specific biases.
* MoE with ~3B active params/token → fits on one A100/H100 (FP8 variant comfortably).
* Emits a **structured** verdict (separate object-correctness and action-correctness
  judgements + an overall verdict + explanation) instead of a single opaque 1-5 score.
  The structured fields land in ``JudgeResult.metadata`` so the benchmark can compute
  object-vs-action error breakdowns and cross-judge agreement.

Evidence: ``mp4_bytes`` — the judge watches the clip. ``judge_stage.py`` withholds the
GT text for ``mp4_bytes`` judges, so the model forms an independent opinion (the prompt
is written for the reference-free case and asks the model to describe what it sees
*before* reading the caption — an anti-confirmation-bias structure that measurably
improved recall for the Gemma4 video judge).

Frame-sampling and resource knobs are read from ``configs/model_registry.yaml`` via
``model_registry`` so they can be tuned without editing this file.
"""

import re
from typing import Any

import torch
from loguru import logger
from PIL import Image

from cosmos_curate.core.interfaces.stage_interface import CuratorStageResource
from cosmos_curate.models import model_registry
from cosmos_curate.models.judge_plugin import (
    EvidenceKind,
    JudgeConfig,
    JudgeItem,
    JudgePlugin,
    JudgeResult,
)
from cosmos_curate.models.judge_video_utils import decode_frames as _decode_frames_shared

# ── Defaults (overridable via configs/model_registry.yaml) ─────────────────────
_DEFAULT_SAMPLING_FPS: float = 2.0
_DEFAULT_MAX_FRAMES: int = 16
_DEFAULT_FRAME_SIZE: int = 448

# ── Prompts ────────────────────────────────────────────────────────────────────
# Structured, anti-confirmation-bias: the model first states what it observes, THEN
# checks the caption against that observation. Output is line-oriented and easy to parse.
_OUTPUT_SPEC = (
    "First, in one short phrase each, state what you actually see:\n"
    "SEEN_OBJECT: <the main object(s) being manipulated, or 'unclear'>\n"
    "SEEN_ACTION: <the action being performed, or 'unclear'>\n"
    "Then judge the caption against what you saw and output exactly these lines:\n"
    "OBJECT: <CORRECT|INCORRECT|UNSURE>\n"
    "ACTION: <CORRECT|INCORRECT|UNSURE>\n"
    "VERDICT: <CORRECT|INCORRECT>\n"
    "EXPLANATION: <one sentence>\n"
    "A clip may contain more than one action (e.g. pick then place). Judge the caption "
    "CORRECT if it faithfully describes any action genuinely present in the clip and does "
    "not assert something that is absent; only mark INCORRECT for a clear contradiction."
)

_QWEN3VL_PROMPTS: dict[str, str] = {
    "structured_grounding_no_gt": (
        "You are a careful, skeptical evaluator of a video caption. Watch the clip and "
        "decide for yourself what object is handled and what action occurs — do not assume "
        "the caption is right.\n\n"
        "Caption to evaluate:\n<caption>{caption_text}</caption>\n\n" + _OUTPUT_SPEC
    ),
    "structured_grounding": (
        "You are a careful, skeptical evaluator of a video caption. Watch the clip and "
        "decide for yourself what object is handled and what action occurs.\n\n"
        "Reference action (may be incomplete — the clip can contain more than this):\n"
        '"{gt_action_text}"\n\n'
        "Caption to evaluate:\n<caption>{caption_text}</caption>\n\n" + _OUTPUT_SPEC
    ),
}
_FALLBACK_PROMPT = _QWEN3VL_PROMPTS["structured_grounding_no_gt"]


def _select_prompt(prompt_variant: str, caption: str, gt_action_text: str) -> str:
    """Pick the structured prompt; prefer the no-GT form when GT is absent."""
    has_gt = bool(gt_action_text and gt_action_text.strip())
    no_gt_key = f"{prompt_variant}_no_gt"
    if not has_gt and no_gt_key in _QWEN3VL_PROMPTS:
        return _QWEN3VL_PROMPTS[no_gt_key].format(caption_text=caption)
    if prompt_variant in _QWEN3VL_PROMPTS:
        return _QWEN3VL_PROMPTS[prompt_variant].format(caption_text=caption, gt_action_text=gt_action_text)
    # Unknown variant → default to the reference-free structured prompt.
    return _FALLBACK_PROMPT.format(caption_text=caption)


def _parse_structured(text: str) -> tuple[str | None, int, str, dict[str, Any]]:
    """Parse the structured output into (verdict, score, explanation, metadata)."""

    def _field(name: str) -> str | None:
        m = re.search(rf"^\s*{name}\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    def _norm(val: str | None) -> str | None:
        if val is None:
            return None
        u = val.strip().upper()
        for tag in ("CORRECT", "INCORRECT", "UNSURE"):
            if u.startswith(tag):
                return tag.lower()
        return None

    verdict_raw = _norm(_field("VERDICT"))
    object_ok = _norm(_field("OBJECT"))
    action_ok = _norm(_field("ACTION"))
    explanation = _field("EXPLANATION") or text.strip()

    # If VERDICT line is missing, derive from object/action sub-verdicts.
    verdict = verdict_raw
    if verdict not in ("correct", "incorrect"):
        if object_ok == "incorrect" or action_ok == "incorrect":
            verdict = "incorrect"
        elif object_ok == "correct" or action_ok == "correct":
            verdict = "correct"
        else:
            verdict = None

    score = 5 if verdict == "correct" else (1 if verdict == "incorrect" else 3)
    metadata = {
        "object_correct": object_ok,
        "action_correct": action_ok,
        "seen_object": _field("SEEN_OBJECT"),
        "seen_action": _field("SEEN_ACTION"),
    }
    return verdict, score, explanation, metadata


class JudgeQwen3VLBase(JudgePlugin):
    """Shared Qwen3-VL judge logic (bf16 and FP8 variants differ only in weights)."""

    def __init__(self) -> None:
        """No I/O at construction — weights load in ``setup``."""
        self._config: JudgeConfig | None = None
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device | None = None
        self._params = model_registry.get_judge_params(self.variant())

    # ── Identity / capability ─────────────────────────────────────────────────

    @property
    def evidence_kind(self) -> EvidenceKind:
        """Watches the clip directly."""
        return "mp4_bytes"

    @property
    def conda_env_name(self) -> str | None:
        """Unified env has torch + a transformers new enough for Qwen3-VL."""
        return str(self._params.get("conda_env", "unified"))

    @property
    def resources(self) -> CuratorStageResource:
        """Read CPU/GPU request from the registry (default 1 full GPU)."""
        res = self._params.get("resources", {}) or {}
        return CuratorStageResource(
            cpus=float(res.get("cpus", 8.0)),
            gpus=float(res.get("gpus", 1.0)),
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self, config: JudgeConfig) -> None:
        """Load the Qwen3-VL model + processor (offline, local weights only)."""
        import os

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        self._config = config

        model_path = self.model_path()
        if model_path is None:
            msg = f"JudgeQwen3VL variant {self.variant()!r} has no model_path."
            raise RuntimeError(msg)

        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"JudgeQwen3VL[{self.variant()}]: loading from {model_path} on {device} ...")

        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            local_files_only=True,
            device_map=None,
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

        self._model = model
        self._processor = processor
        self._device = device
        logger.info(f"JudgeQwen3VL[{self.variant()}]: ready on {device}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _decode(self, mp4_bytes: bytes) -> list[Image.Image]:
        return _decode_frames_shared(
            mp4_bytes,
            sampling_fps=float(self._params.get("sampling_fps", _DEFAULT_SAMPLING_FPS)),
            max_frames=int(self._params.get("max_frames", _DEFAULT_MAX_FRAMES)),
            frame_size=int(self._params.get("frame_size", _DEFAULT_FRAME_SIZE)),
        )

    def _build_text(self, frames: list[Image.Image], caption: str, gt_action_text: str) -> str:
        assert self._processor is not None
        assert self._config is not None
        prompt = _select_prompt(self._config.prompt_variant, caption, gt_action_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def judge_batch(self, items: list[JudgeItem]) -> list[JudgeResult]:
        """Run true batched GPU inference: one processor call + one generate() per batch."""
        assert self._model is not None
        assert self._processor is not None
        assert self._device is not None
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
                frames = self._decode(item.mp4_bytes)
                if not frames:
                    results[i] = JudgeResult(verdict=None, score=1, explanation="0 frames decoded")
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"JudgeQwen3VL[{self.variant()}] frame decode failed: {exc}")
                results[i] = JudgeResult(verdict=None, score=1, explanation=str(exc))
                continue
            valid_indices.append(i)
            valid_frames.append(frames)
            valid_texts.append(self._build_text(frames, item.caption, item.gt_action_text or ""))

        if not valid_indices:
            return [r or JudgeResult(verdict=None, score=1, explanation="skipped") for r in results]

        try:
            self._processor.tokenizer.padding_side = "left"
            inputs = self._processor(
                text=valid_texts,
                videos=valid_frames,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._config.max_new_tokens,
                    do_sample=False,
                )

            new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
            decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)

            for orig_i, raw in zip(valid_indices, decoded, strict=False):
                verdict, score, explanation, metadata = _parse_structured(raw)
                results[orig_i] = JudgeResult(
                    verdict=verdict,
                    score=score,
                    explanation=explanation,
                    raw_output=raw,
                    metadata=metadata,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"JudgeQwen3VL[{self.variant()}] batch inference failed: {exc}")
            for orig_i in valid_indices:
                if results[orig_i] is None:
                    results[orig_i] = JudgeResult(verdict=None, score=1, explanation=str(exc))

        return [r or JudgeResult(verdict=None, score=1, explanation="unknown error") for r in results]


class JudgeQwen3VL30B(JudgeQwen3VLBase):
    """Qwen3-VL-30B-A3B-Instruct (bf16). ~60 GB weights; MoE, ~3B active/token."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "qwen3vl_30b"


class JudgeQwen3VL30BFP8(JudgeQwen3VLBase):
    """Qwen3-VL-30B-A3B-Instruct-FP8. ~half the VRAM; fits one A100/H100."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "qwen3vl_30b_fp8"
