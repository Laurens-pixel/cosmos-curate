# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase wrapper for the evaluation/judging stage.

Goes after ``CaptioningPhase`` and before ``OutputPhase``. Only added to the pipeline
when the user passes ``--evaluate``.
"""

import attrs

from cosmos_curate.core.interfaces.phase_interface import CurationPhase
from cosmos_curate.core.interfaces.stage_interface import CuratorStage, CuratorStageSpec
from cosmos_curate.pipelines.video.evaluation.gt_sources import get_gt_source_class
from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource
from cosmos_curate.pipelines.video.evaluation.judge_stage import JudgeStage, JudgeStageConfig


@attrs.define(frozen=True)
class JudgePhaseConfig:
    """User-facing configuration for the JudgePhase."""

    judge_variant: str
    caption_source: str

    gt_source: str = "agibot"
    gt_source_kwargs: dict[str, object] = attrs.Factory(dict)
    """Constructor kwargs forwarded to the GT source (e.g. ``task_info_dir``)."""

    prompt_variant: str = "lenient_binary"
    prompt_text: str | None = None

    batch_size: int = 8
    max_new_tokens: int = 32
    extra: dict[str, object] = attrs.Factory(dict)

    verbose: bool = False
    perf_profile: bool = False


def _build_gt_source(name: str, kwargs: dict[str, object]) -> GtSource:
    cls = get_gt_source_class(name)
    return cls(**kwargs)  # type: ignore[arg-type]


class JudgePhase(CurationPhase):
    """Run a single judge over the captions populated upstream."""

    def __init__(self, config: JudgePhaseConfig) -> None:
        """Validate and remember the config."""
        self._cfg = config

    @property
    def name(self) -> str:
        """Return the phase name."""
        return f"judge:{self._cfg.judge_variant}"

    @property
    def requires(self) -> frozenset[str]:
        """JudgePhase consumes captions populated by CaptioningPhase."""
        return frozenset({"captioned"})

    @property
    def populates(self) -> frozenset[str]:
        """Mark windows as judged so downstream phases can declare a dependency."""
        return frozenset({"judged"})

    def build_stages(self) -> list[CuratorStage | CuratorStageSpec]:
        """Build the single JudgeStage for this judge variant."""
        cfg = self._cfg
        gt_source = _build_gt_source(cfg.gt_source, dict(cfg.gt_source_kwargs))
        stage_cfg = JudgeStageConfig(
            judge_variant=cfg.judge_variant,
            caption_source=cfg.caption_source,
            prompt_variant=cfg.prompt_variant,
            prompt_text=cfg.prompt_text,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
            extra=dict(cfg.extra),
            verbose=cfg.verbose,
            log_stats=cfg.perf_profile,
        )
        return [CuratorStageSpec(JudgeStage(stage_cfg, gt_source), num_workers_per_node=1)]
