# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic JudgeStage that delegates to a JudgePlugin.

This stage is judge-agnostic — adding a new judge requires only a new ``JudgePlugin``
implementation and a registry entry; this file is never modified.
"""

from typing import Any

import attrs
import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curate.core.interfaces.stage_interface import (
    CuratorStage,
    CuratorStageResource,
    PipelineTask,
)
from cosmos_curate.core.utils.infra.performance_utils import StageTimer
from cosmos_curate.models.judge_interface import make_judge_plugin
from cosmos_curate.models.judge_plugin import JudgeConfig, JudgeItem, JudgeResult
from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource
from cosmos_curate.pipelines.video.evaluation.prompts import get_prompt
from cosmos_curate.pipelines.video.utils.data_model import (
    SplitPipeTask,
    Window,
    get_video_from_task,
)


@attrs.define(frozen=True)
class JudgeStageConfig:
    """Configuration the JudgeStage needs at construction time.

    The plugin variant + prompt name are all that's required to identify a
    behaviour; everything else has sensible defaults.
    """

    judge_variant: str
    """Name of the JudgePlugin to use (e.g. ``gemma4_e4b``)."""

    caption_source: str
    """Which key in ``Window.caption`` to judge (e.g. ``qwen``, ``gemma4``)."""

    prompt_variant: str = "lenient_binary"
    """Prompt template name (see ``evaluation/prompts.py``)."""

    prompt_text: str | None = None
    """Optional override prompt text. ``None`` => use ``prompt_variant``."""

    batch_size: int = 8
    max_new_tokens: int = 32
    extra: dict[str, Any] = attrs.Factory(dict)

    verbose: bool = False
    log_stats: bool = True


class JudgeStage(CuratorStage):
    """A pluggable caption-judging stage.

    For each window in the task it:
      1. Reads ``window.caption[caption_source]``.
      2. Looks up GT via the configured ``GtSource``.
      3. Builds a ``JudgeItem`` and forwards it to the plugin's ``judge_batch``.
      4. Writes the result back to ``window.judge[plugin.variant()]``.
    """

    def __init__(self, config: JudgeStageConfig, gt_source: GtSource) -> None:
        """Construct the stage.

        Note: the plugin itself is *not* instantiated here — that happens in
        ``stage_setup`` so it runs in the right Ray worker / conda env.
        """
        super().__init__()
        self._timer = StageTimer(self)
        self._config = config
        self._gt_source = gt_source
        # Resolve the plugin class up-front to expose static metadata
        # (resources, conda_env_name) at construction time without loading the model.
        from cosmos_curate.models.judge_interface import get_judge_plugin_class

        self._plugin_cls = get_judge_plugin_class(config.judge_variant)
        self._plugin = None  # type: ignore[var-annotated]

    @property
    def resources(self) -> CuratorStageResource:
        """Forward the plugin's resource requirements."""
        # We instantiate to read resources; harmless because plugin __init__ is light.
        if self._plugin is None:
            self._plugin = self._plugin_cls()
        return self._plugin.resources

    @property
    def conda_env_name(self) -> str | None:
        """Forward the plugin's conda env name."""
        if self._plugin is None:
            self._plugin = self._plugin_cls()
        return self._plugin.conda_env_name

    def secondary_name(self) -> str:
        """Append the variant to the stage name in logs/profiles."""
        return self._config.judge_variant

    def stage_setup(self) -> None:
        """Instantiate and warm up the plugin in the worker process."""
        if self._plugin is None:
            self._plugin = self._plugin_cls()
        prompt = self._config.prompt_text or get_prompt(self._config.prompt_variant)
        config = JudgeConfig(
            variant=self._config.judge_variant,
            batch_size=self._config.batch_size,
            max_new_tokens=self._config.max_new_tokens,
            prompt=prompt,
            prompt_variant=self._config.prompt_variant,
            extra=dict(self._config.extra),
        )
        self._plugin.setup(config)

    # ── Per-task processing ───────────────────────────────────────────────────

    def _build_items_for_video(
        self, video: Any  # noqa: ANN401 — Video defined in data_model
    ) -> tuple[list[JudgeItem], list[tuple[Window, str]]]:
        """Walk the video's windows and produce parallel lists of items + back-pointers.

        ``items[i]`` is the JudgeItem to send to the plugin; ``back[i]`` is
        ``(window, gt_action_text)`` so we can write the result back and stash GT
        on the saved record without consulting the GT source again.
        """
        items: list[JudgeItem] = []
        back: list[tuple[Window, str]] = []
        variant = self._config.judge_variant
        cap_source = self._config.caption_source
        evidence_kind = self._plugin.evidence_kind  # type: ignore[union-attr]

        for clip in video.clips:
            for window in clip.windows:
                # Skip if already judged (resume support)
                if variant in window.judge:
                    continue
                caption = window.caption.get(cap_source, "")
                if not caption:
                    continue

                video_path_str = str(video.input_video)
                gt_text, gt_extras = self._gt_source.lookup(
                    video_path_str,
                    window.start_frame,
                    window.end_frame,
                )

                # Video-evidence judges (mp4_bytes) can watch the clip directly —
                # withhold GT text so they form an independent opinion.
                item_gt = gt_text if evidence_kind == "text" else None

                item = JudgeItem(
                    caption=caption,
                    gt_action_text=item_gt,
                    metadata={
                        "source_video": video_path_str,
                        "clip_uuid": str(clip.uuid),
                        "window_key": f"{window.start_frame}_{window.end_frame}",
                        "start_frame": window.start_frame,
                        "end_frame": window.end_frame,
                        "gt_extras": gt_extras,
                    },
                )

                # Attach evidence on demand
                if evidence_kind == "mp4_bytes":
                    mp4 = window.mp4_bytes.resolve() if window.mp4_bytes is not None else None
                    if mp4 is not None:
                        item.mp4_bytes = bytes(mp4)
                # ``frames`` evidence is added by future plugins that decode frames in
                # their ``judge_batch`` from window.mp4_bytes — keeping decode-once logic
                # local to those plugins avoids decoding overhead for text-only judges.

                items.append(item)
                back.append((window, gt_text))

        return items, back

    def _record_result(self, window: Window, gt_text: str, item: JudgeItem, result: JudgeResult) -> None:
        """Write a single judge result onto the window."""
        record: dict[str, Any] = {
            "verdict": result.verdict,
            "score": result.score,
            "explanation": result.explanation,
            "raw_output": result.raw_output,
            "gt_action_text": gt_text,
            "caption_source": self._config.caption_source,
            "prompt_variant": self._config.prompt_variant,
        }
        gt_extras = item.metadata.get("gt_extras") or {}
        if gt_extras:
            record["gt_extras"] = dict(gt_extras)
        if result.metadata:
            record["plugin_metadata"] = dict(result.metadata)
        window.judge[self._config.judge_variant] = record

    @nvtx.annotate("JudgeStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask]:
        """Judge every captioned window in every task."""
        if self._plugin is None:
            msg = "stage_setup() must be called before process_data()."
            raise RuntimeError(msg)

        for task in tasks:
            major_size = task.get_major_size()
            self._timer.reinit(self, major_size)
            video = get_video_from_task(task)

            with self._timer.time_process():
                items, back = self._build_items_for_video(video)
                if not items:
                    if self._config.verbose:
                        logger.debug(
                            f"JudgeStage[{self._config.judge_variant}]: nothing to judge for {video.input_video}"
                        )
                    continue

                bs = max(1, self._config.batch_size)
                for i in range(0, len(items), bs):
                    chunk = items[i : i + bs]
                    chunk_back = back[i : i + bs]
                    try:
                        results = self._plugin.judge_batch(chunk)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            f"JudgeStage[{self._config.judge_variant}] batch failed at {i}: {exc}"
                        )
                        results = [
                            JudgeResult(verdict=None, score=1, explanation=str(exc)) for _ in chunk
                        ]

                    if len(results) != len(chunk):
                        msg = (
                            f"Plugin returned {len(results)} results for batch of "
                            f"{len(chunk)} items — plugins must preserve length."
                        )
                        raise RuntimeError(msg)

                    for (window, gt_text), item, result in zip(chunk_back, chunk, results, strict=True):
                        self._record_result(window, gt_text, item, result)

            stage_perf = getattr(task, "stage_perf", None)
            if self._config.log_stats and stage_perf is not None:
                stage_name, stage_perf_stats = self._timer.log_stats()
                stage_perf[stage_name] = stage_perf_stats

        return tasks

    # CuratorStage protocol — keep base behaviour
    def process(self, tasks: list[PipelineTask]) -> list[PipelineTask]:  # type: ignore[override]
        """Delegate to ``process_data`` after type narrowing."""
        return self.process_data([t for t in tasks if isinstance(t, SplitPipeTask)])  # type: ignore[arg-type]
