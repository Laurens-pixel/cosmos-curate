# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 text-only judge plugins (E4B-it and 31B-it).

Mirrors the ``vllm_qwen.py`` shape: a shared base class implements the heavy lifting,
with thin subclasses overriding ``variant()`` and any size-specific config.

Inputs are caption + GT text; the model is invoked via HuggingFace transformers
(no vLLM, no video). Output is parsed into a binary ``correct`` / ``incorrect``
verdict; the lenient prompt is the default and matches the best-performing
configuration from the offline evaluation (F1 = 0.859 on the 30-vid manual GT
benchmark, the highest of any judge tested).
"""

import os
import re
import sys
from typing import Any

import torch
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import CuratorStageResource
from cosmos_curate.models.judge_plugin import (
    EvidenceKind,
    JudgeConfig,
    JudgeItem,
    JudgePlugin,
    JudgeResult,
)


# ── Shared constants ─────────────────────────────────────────────────────────

# Common pip_overrides path inside the container — Gemma 4 needs transformers >= 5.5.0.
_PIP_OVERRIDES = "/config/pip_overrides"
# Unified env site-packages — needed because Ray workers don't auto-include them.
_UNIFIED_SP = "/opt/cosmos-curate/.pixi/envs/unified/lib/python3.12/site-packages"


def _patch_hf_type_validator() -> None:
    """Make huggingface_hub.dataclasses.type_validator understand PEP 604 unions (str | None).

    huggingface_hub 0.36.0 (in the container) only handles typing.Union, not the newer
    types.UnionType produced by the X | Y syntax.  Patching the module-level function is
    sufficient because _create_type_validator uses a late-binding closure that re-looks up
    type_validator in the module globals on every call.
    """
    import types as _types

    try:
        import huggingface_hub.dataclasses as _hf_dc
    except ImportError:
        return
    if getattr(_hf_dc, "_pep604_patched", False):
        return
    _orig = _hf_dc.type_validator

    def _patched(name: str, value: object, expected_type: object) -> None:
        if isinstance(expected_type, _types.UnionType):
            for t in expected_type.__args__:  # type: ignore[union-attr]
                try:
                    _patched(name, value, t)
                    return
                except TypeError:
                    pass
            raise TypeError(f"Field '{name}' value {value!r} doesn't match {expected_type}")
        _orig(name, value, expected_type)

    _hf_dc.type_validator = _patched
    _hf_dc._pep604_patched = True  # type: ignore[attr-defined]


def _ensure_imports() -> None:
    """Make sure transformers 5.5.0 + tokenizers are importable from this worker."""
    if _UNIFIED_SP not in sys.path:
        sys.path.insert(0, _UNIFIED_SP)
    if _PIP_OVERRIDES not in sys.path:
        sys.path.insert(0, _PIP_OVERRIDES)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    _patch_hf_type_validator()


def _parse_verdict(text: str) -> tuple[str | None, str]:
    """Return ``(verdict, explanation)`` parsed from a generative model output.

    The output convention is: first non-empty line carries the verdict word, remaining
    lines form the explanation.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    first = lines[0].upper() if lines else ""
    explanation = " ".join(lines[1:]) if len(lines) > 1 else text.strip()

    # Order matters — INCORRECT_OBJECT/INCORRECT_ACTION must come before bare INCORRECT.
    patterns = (
        ("incorrect_object", r"INCORRECT_OBJECT"),
        ("incorrect_action", r"INCORRECT_ACTION"),
        ("incorrect", r"\bINCORRECT\b"),
        ("correct", r"\bCORRECT\b"),
    )
    for label, pat in patterns:
        if re.search(pat, first):
            return label, explanation

    upper_text = text.upper()
    for label, pat in patterns:
        if re.search(pat, upper_text):
            return label, text.strip()

    return None, text.strip()


def _verdict_to_score(verdict: str | None) -> int:
    """Map verdict to the 5/1 score format used by ``check_scores.py``."""
    return 5 if verdict == "correct" else 1


# ── Base class ───────────────────────────────────────────────────────────────


class JudgeGemma4Base(JudgePlugin):
    """Shared text-only Gemma 4 judge.

    Subclasses override ``variant()`` and may override ``resources`` /
    ``_load_kwargs`` for size-specific behaviour.
    """

    # Subclass overrides (kept as class vars for easy configuration)
    _DEFAULT_BATCH_SIZE: int = 8

    def __init__(self) -> None:
        """No I/O at construction — model loading happens in ``setup``."""
        self._config: JudgeConfig | None = None
        self._model: Any = None
        self._processor: Any = None
        self._input_device: torch.device | None = None
        self._prompt_template: str = ""

    # ── Static metadata ───────────────────────────────────────────────────────

    @property
    def evidence_kind(self) -> EvidenceKind:
        """This is a text-only judge."""
        return "text"

    @property
    def conda_env_name(self) -> str | None:
        """Run inside the container's unified env (has torch + transformers)."""
        return "unified"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self, config: JudgeConfig) -> None:
        """Load the Gemma 4 model and processor."""
        _ensure_imports()

        self._config = config
        self._prompt_template = config.prompt or ""
        if not self._prompt_template:
            msg = "JudgeGemma4: prompt is required (pass via JudgeConfig.prompt)."
            raise ValueError(msg)

        model_path = self.model_path()
        if model_path is None:
            msg = f"JudgeGemma4 variant {self.variant()!r} has no model_path."
            raise RuntimeError(msg)

        from transformers import AutoProcessor

        logger.info(f"JudgeGemma4[{self.variant()}]: loading model from {model_path} ...")

        load_kwargs = self._load_kwargs()
        from transformers import Gemma4ForConditionalGeneration

        model = Gemma4ForConditionalGeneration.from_pretrained(  # type: ignore[no-untyped-call]
            str(model_path),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **load_kwargs,
        ).eval()
        if "device_map" not in load_kwargs:
            # Single-GPU placement
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            input_device = device
        else:
            # device_map="auto" => model is already sharded; figure out where embeddings live
            try:
                input_device = next(model.get_input_embeddings().parameters()).device
            except Exception:  # noqa: BLE001
                input_device = next(model.parameters()).device

        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._model = model
        self._processor = processor
        self._input_device = input_device
        logger.info(f"JudgeGemma4[{self.variant()}]: ready on {input_device}")

    def _load_kwargs(self) -> dict[str, Any]:
        """Subclasses override to inject ``device_map``, quantization, etc."""
        return {}

    @property
    def resources(self) -> CuratorStageResource:
        """Default: 1 GPU. Subclasses override for their specific size."""
        return CuratorStageResource(cpus=1.0, gpus=1.0)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_batch(self, prompts: list[str]) -> list[str]:
        """Greedy-decode a batch of prompts."""
        assert self._processor is not None
        assert self._model is not None
        assert self._input_device is not None
        assert self._config is not None

        messages_list = [
            [{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts
        ]
        texts = [
            self._processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self._input_device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return list(self._processor.batch_decode(new_tokens, skip_special_tokens=True))

    def judge_batch(self, items: list[JudgeItem]) -> list[JudgeResult]:
        """Render the prompt for each item, run greedy decoding, parse the verdict."""
        prompts = [
            self._prompt_template.format(gt_action_text=it.gt_action_text, caption_text=it.caption)
            for it in items
        ]
        try:
            raws = self._run_batch(prompts)
        except Exception as exc:
            logger.exception(f"JudgeGemma4[{self.variant()}] batch failed: {exc}")
            raws = [""] * len(items)

        results: list[JudgeResult] = []
        for raw in raws:
            verdict, explanation = _parse_verdict(raw)
            results.append(
                JudgeResult(
                    verdict=verdict,
                    score=_verdict_to_score(verdict),
                    explanation=explanation,
                    raw_output=raw,
                )
            )
        return results


# ── Concrete plugins ────────────────────────────────────────────────────────


class JudgeGemma4E4B(JudgeGemma4Base):
    """Gemma 4 E4B-it (~8 B params, single A100). Default text judge."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "gemma4_e4b"

    @property
    def resources(self) -> CuratorStageResource:
        """0.5 GPU + 8 CPUs. Worker count is capped to 1 via CuratorStageSpec in
        JudgePhase.build_stages() so only one 9 GB model copy is ever loaded."""
        return CuratorStageResource(cpus=8.0, gpus=0.5)


class JudgeGemma4_31B(JudgeGemma4Base):
    """Gemma 4 31B-it. Requires sharding across two GPUs (62 GB > one A100)."""

    @staticmethod
    def variant() -> str:
        """Return the unique judge identifier."""
        return "gemma4_31b"

    @property
    def resources(self) -> CuratorStageResource:
        """Sharded across two A100s via ``device_map='auto'``."""
        return CuratorStageResource(cpus=2.0, gpus=2.0)

    def _load_kwargs(self) -> dict[str, Any]:
        """Use ``device_map='auto'`` so accelerate places layers across both GPUs."""
        return {"device_map": "auto"}
