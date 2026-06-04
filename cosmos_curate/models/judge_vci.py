# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VCInspector judge plugins (3B and 7B).

VC-Inspector is a LoRA-tuned Qwen2.5-VL model trained on ActivityNet that scores
captions 1–5 for factual accuracy (object + action correctness).

Unlike the text-only Gemma4 judge, VCInspector is a vision-language model that
receives decoded video frames alongside the caption, making it evidence_kind="mp4_bytes".
The plugin decodes frames from the window's mp4 bytes via PyAV before inference.

Reference: https://huggingface.co/dipta007/VCInspector-7B
Offline evaluation results (30 videos, 125 windows, tasks 327/352/354):
  - 3B: P=70.7% R=46.0% F1=0.558  Acc=63.2%
  - 7B: P=82.0% R=65.1% F1=0.726  Acc=75.2%  ← best single judge on Qwen captions
"""

import re
from typing import Any

import torch
from loguru import logger
from PIL import Image

from cosmos_curate.core.interfaces.stage_interface import CuratorStageResource
from cosmos_curate.models.judge_plugin import (
    EvidenceKind,
    JudgeConfig,
    JudgeItem,
    JudgePlugin,
    JudgeResult,
)
from cosmos_curate.models.judge_video_utils import decode_frames as _decode_frames_shared

# ── Constants ────────────────────────────────────────────────────────────────

_SAMPLING_FPS: float = 1.0   # VCInspector training default
_MAX_FRAMES: int = 32         # "uniformly sampled into 32 frames" (paper)
_FRAME_SIZE: int = 224        # "resized to 224×224" (paper); VIDEO_MAX_PIXELS=50176

_SCORE_INSTRUCTIONS = (
    "Please rate the helpfulness, relevance, accuracy, level of details of the caption. "
    "The overall score should be on a scale of 1 to 5, where a higher score indicates "
    "better overall performance. Please first output a single line containing only one "
    "integer indicating the score. In the subsequent line, please provide a comprehensive "
    "explanation of your evaluation, avoiding any potential bias. STRICTLY FOLLOW THE FORMAT."
)

# Prompt templates keyed by judge-prompt-variant name.
# All templates accept {caption_text} and {gt_action_text} placeholders.
_VCI_PROMPTS: dict[str, str] = {
    # --- Robot manipulation ---
    "lenient_binary": (
        "<caption>{caption_text}</caption>\n\n"
        "Ground truth action: {gt_action_text}\n\n"
        "This video shows a robotic arm performing a manipulation task. "
        "Focus primarily on whether the caption correctly identifies the overall action goal "
        "(e.g. retrieve, place, push, grasp). "
        "Object naming is secondary — a vague description, a visual description, or a synonym "
        "is acceptable (e.g. 'purple vegetable' for 'onion', 'bottle' for 'green tea bottle'). "
        "Only penalise heavily if the caption clearly names a different object or describes "
        "the opposite action (e.g. placing when the GT is retrieving). "
        + _SCORE_INSTRUCTIONS
    ),
    "lenient_binary_no_gt": (
        "<caption>{caption_text}</caption>\n\n"
        "This video shows a robotic arm performing a manipulation task. "
        "Focus primarily on whether the caption correctly identifies the overall action goal "
        "(e.g. retrieve, place, push, grasp). "
        "Object naming is secondary — vague or approximate descriptions are acceptable. "
        "Only penalise heavily if the caption describes the opposite action or a completely "
        "unrelated activity. "
        + _SCORE_INSTRUCTIONS
    ),
    # --- Autonomous driving / AV (nuScenes domain) ---
    "av_binary": (
        "<caption>{caption_text}</caption>\n\n"
        "Ground truth scene tags: {gt_action_text}\n\n"
        "This video is from a camera mounted on an autonomous vehicle. "
        "Check whether the caption is consistent with the driving scene described by the GT tags. "
        "Be lenient on camera angle, perspective, and emphasis — the caption may focus on a subset. "
        "Only penalise if the caption clearly contradicts the GT (e.g. indoor vs outdoor, "
        "parking lot vs highway, night vs day) or describes a completely different set of objects. "
        + _SCORE_INSTRUCTIONS
    ),
    "av_binary_no_gt": (
        "<caption>{caption_text}</caption>\n\n"
        "This video is from a camera mounted on an autonomous vehicle. "
        "Rate how accurately this caption describes the driving scene shown in the video. "
        "Focus on whether the road type, key objects (vehicles, pedestrians, infrastructure), "
        "and environmental conditions (day/night, weather) are correctly described. "
        "Only penalise heavily for captions that are clearly implausible for a driving scene "
        "or contain obvious factual contradictions. "
        + _SCORE_INSTRUCTIONS
    ),
    # --- Industrial assembly (InHARD domain) ---
    "inhard_binary": (
        "<caption>{caption_text}</caption>\n\n"
        "Ground truth action label: {gt_action_text}\n\n"
        "This video shows a factory worker performing an industrial assembly task. "
        "Your default should be CORRECT (score 5) unless you see a clear contradiction. "
        "Be very lenient: 'picks up' and 'Take' are the same; 'sets down', 'places', 'puts' and "
        "'Put down' are the same; any vague object description ('a tool', 'an object', 'a part', "
        "'something') is acceptable; describing extra context or adjacent steps is fine. "
        "Many clips are under 1 second — if only a hand or object is briefly visible and the "
        "caption is a plausible description of that view, score it 5. "
        "Only give a low score (1–3) if the caption explicitly names the wrong action direction "
        "(e.g. placing when the worker is clearly picking up) OR names a clearly wrong specific "
        "object (e.g. 'screwdriver' when a measuring rod is visibly held). "
        + _SCORE_INSTRUCTIONS
    ),
    "inhard_binary_no_gt": (
        "<caption>{caption_text}</caption>\n\n"
        "This video shows a factory worker performing an industrial assembly task. "
        "Your default should be CORRECT (score 5) unless you see a clear contradiction. "
        "Many clips are under 1 second — a brief glimpse of a hand, a tool, or a component "
        "is all that may be visible, so reward captions that make a reasonable attempt. "
        "Vague descriptions ('picks up an object', 'holds a tool', 'reaches for a part') are "
        "acceptable. Only give a low score (1–3) if the caption makes an explicit claim that "
        "directly contradicts what is clearly visible in the video. "
        + _SCORE_INSTRUCTIONS
    ),
    # --- Cooking (YouCook2) ---
    "youcook2_binary": (
        "<caption>{caption_text}</caption>\n\n"
        "Ground truth action: {gt_action_text}\n\n"
        "This video shows a cooking step from a recipe. "
        "Focus primarily on whether the caption correctly identifies the cooking action being performed "
        "(e.g. chopping, boiling, frying, mixing). "
        "Ingredient naming is secondary — a vague description, a synonym, or omitting a specific "
        "ingredient is acceptable as long as the action type is correct. "
        "Only penalise heavily if the caption describes a completely different cooking action. "
        + _SCORE_INSTRUCTIONS
    ),
    "youcook2_binary_no_gt": (
        "<caption>{caption_text}</caption>\n\n"
        "This video shows a cooking step from a recipe. "
        "Rate how accurately this caption describes what is happening in the video. "
        "Focus on whether the cooking action (e.g. chopping, boiling, frying, mixing) is correct. "
        "Ingredient naming is secondary — approximate or vague ingredient descriptions are acceptable. "
        + _SCORE_INSTRUCTIONS
    ),
}

# Fallback for unknown variants — agibot domain (backward compatible).
_PROMPT_TEMPLATE = _VCI_PROMPTS["lenient_binary"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _decode_frames(mp4_bytes: bytes) -> list[Image.Image]:
    """Decode up to _MAX_FRAMES frames from raw MP4 bytes using VCI paper parameters."""
    return _decode_frames_shared(
        mp4_bytes,
        sampling_fps=_SAMPLING_FPS,
        max_frames=_MAX_FRAMES,
        frame_size=_FRAME_SIZE,
    )


def _parse_score(text: str) -> tuple[int | None, str]:
    """Extract integer score 1–5 from first matching line; rest is explanation."""
    lines = text.strip().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*([1-5])\s*$", line.strip())
        if m:
            return int(m.group(1)), "\n".join(lines[i + 1 :]).strip()
    # Fallback: find a lone digit 1-5 anywhere in first two lines
    for line in lines[:2]:
        m = re.search(r"\b([1-5])\b", line)
        if m:
            return int(m.group(1)), text.strip()
    return None, text.strip()


def _score_to_verdict(score: int | None) -> str | None:
    if score is None:
        return None
    return "correct" if score == 5 else "incorrect"


# ── Base plugin ──────────────────────────────────────────────────────────────


class JudgeVCIBase(JudgePlugin):
    """Shared VCInspector judge logic.

    Subclasses override ``variant()`` and ``resources`` for 3B vs 7B.
    """

    def __init__(self) -> None:
        """No I/O at construction — model loading happens in ``setup``."""
        self._config: JudgeConfig | None = None
        self._model: Any = None
        self._processor: Any = None
        self._device: torch.device | None = None

    # ── Identity / capability ─────────────────────────────────────────────────

    @property
    def evidence_kind(self) -> EvidenceKind:
        """VCInspector needs the window mp4 bytes to decode frames."""
        return "mp4_bytes"

    @property
    def conda_env_name(self) -> str | None:
        """unified env has torch + transformers with Qwen2.5-VL support."""
        return "unified"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self, config: JudgeConfig) -> None:
        """Load the VCInspector model and processor."""
        import os

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        self._config = config

        model_path = self.model_path()
        if model_path is None:
            msg = f"JudgeVCI variant {self.variant()!r} has no model_path."
            raise RuntimeError(msg)

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"JudgeVCI[{self.variant()}]: loading from {model_path} on {device} ...")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.float16,
            local_files_only=True,
        ).to(device)
        model.eval()

        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

        self._model = model
        self._processor = processor
        self._device = device
        logger.info(f"JudgeVCI[{self.variant()}]: ready on {device}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer_single(self, frames: list[Image.Image], caption: str, gt_action_text: str = "") -> str:
        """Run one inference call; returns the raw decoded string."""
        assert self._model is not None
        assert self._processor is not None
        assert self._device is not None
        assert self._config is not None

        # Prefer the _no_gt variant so VCI is always reference-free.
        # Only fall back to the GT-bearing template if a _no_gt sibling doesn't exist.
        no_gt_variant = self._config.prompt_variant + "_no_gt"
        if no_gt_variant in _VCI_PROMPTS:
            template = _VCI_PROMPTS[no_gt_variant]
            prompt = template.format(caption_text=caption)
        else:
            template = _VCI_PROMPTS.get(self._config.prompt_variant, _PROMPT_TEMPLATE)
            prompt = template.format(caption_text=caption, gt_action_text=gt_action_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            videos=[frames],
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

        new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _build_text(self, frames: list[Image.Image], caption: str, gt_action_text: str = "") -> str:
        """Build the formatted prompt text for one item (frames required for correct token count)."""
        assert self._processor is not None
        no_gt_variant = self._config.prompt_variant + "_no_gt"  # type: ignore[union-attr]
        if no_gt_variant in _VCI_PROMPTS:
            template = _VCI_PROMPTS[no_gt_variant]
            prompt = template.format(caption_text=caption)
        else:
            template = _VCI_PROMPTS.get(self._config.prompt_variant, _PROMPT_TEMPLATE)  # type: ignore[union-attr]
            prompt = template.format(caption_text=caption, gt_action_text=gt_action_text)
        messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]}]
        return self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def judge_batch(self, items: list[JudgeItem]) -> list[JudgeResult]:
        """True batched GPU inference: one processor call + one model.generate() for all items."""
        assert self._model is not None
        assert self._processor is not None
        assert self._device is not None
        assert self._config is not None

        results: list[JudgeResult | None] = [None] * len(items)
        valid_indices: list[int] = []
        valid_frames: list[list[Image.Image]] = []
        valid_texts: list[str] = []

        # Decode frames and build prompts for each valid item
        for i, item in enumerate(items):
            if item.mp4_bytes is None:
                results[i] = JudgeResult(verdict=None, score=1, explanation="no mp4_bytes")
                continue
            try:
                frames = _decode_frames(item.mp4_bytes)
                if not frames:
                    results[i] = JudgeResult(verdict=None, score=1, explanation="0 frames decoded")
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"JudgeVCI[{self.variant()}] frame decode failed: {exc}")
                results[i] = JudgeResult(verdict=None, score=1, explanation=str(exc))
                continue
            valid_indices.append(i)
            valid_frames.append(frames)
            valid_texts.append(self._build_text(frames, item.caption, item.gt_action_text or ""))

        if not valid_indices:
            return [r or JudgeResult(verdict=None, score=1, explanation="skipped") for r in results]

        try:
            # Left-pad so all sequences are aligned for generation
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
                    temperature=0.0,
                    do_sample=False,
                )

            # Slice off the (left-padded) input prefix — same length for all items
            new_tokens = generated_ids[:, inputs["input_ids"].shape[1] :]
            decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)

            for orig_i, raw in zip(valid_indices, decoded):
                score, explanation = _parse_score(raw)
                results[orig_i] = JudgeResult(
                    verdict=_score_to_verdict(score),
                    score=score if score is not None else 1,
                    explanation=explanation,
                    raw_output=raw,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"JudgeVCI[{self.variant()}] batch inference failed: {exc}")
            for orig_i in valid_indices:
                if results[orig_i] is None:
                    results[orig_i] = JudgeResult(verdict=None, score=1, explanation=str(exc))

        return [r or JudgeResult(verdict=None, score=1, explanation="unknown error") for r in results]


# ── Concrete plugins ──────────────────────────────────────────────────────────


class JudgeVCI3B(JudgeVCIBase):
    """VCInspector-3B (LoRA-tuned Qwen2.5-VL-3B). ~7.5 GB VRAM."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "vci_3b"

    @property
    def resources(self) -> CuratorStageResource:
        """Single A100 is sufficient for the 3B model."""
        return CuratorStageResource(cpus=1.0, gpus=1.0)


class JudgeVCI7B(JudgeVCIBase):
    """VCInspector-7B (LoRA-tuned Qwen2.5-VL-7B). ~16 GB VRAM. Best F1 of all judges tested."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "vci_7b"

    @property
    def resources(self) -> CuratorStageResource:
        """0.5 GPU + 8 CPUs. Worker count is capped to 1 via CuratorStageSpec in
        JudgePhase.build_stages() so only one 16 GB model copy is ever loaded."""
        return CuratorStageResource(cpus=8.0, gpus=0.5)
