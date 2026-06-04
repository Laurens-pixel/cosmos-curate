# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Judge plugin definition.

This interface defines the contract for adding new caption-judge models to cosmos-curate.

To add a new judge:
1. Create a class inheriting from ``JudgePlugin``
2. Implement the abstract methods below
3. Register in ``cosmos_curate/models/judge_interface.py:_JUDGE_PLUGINS``
4. Add model ID mapping (if applicable) in ``cosmos_curate/models/judge_model_ids.py``

References:
- VLLM_INTERFACE_PLUGIN.md (sibling pattern for vLLM models)

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import attrs

from cosmos_curate.core.interfaces.stage_interface import CuratorStageResource
from cosmos_curate.core.utils.model import model_utils
from cosmos_curate.models.judge_model_ids import get_judge_model_id

if TYPE_CHECKING:
    import torch

EvidenceKind = Literal["text", "frames", "mp4_bytes"]


@attrs.define(frozen=True)
class JudgeConfig:
    """Runtime configuration for a JudgePlugin worker."""

    variant: str
    """Plugin variant identifier (e.g. ``gemma4_e4b``, ``vci_7b``)."""

    batch_size: int = 8
    """How many ``JudgeItem`` objects to send through ``judge_batch`` per call."""

    max_new_tokens: int = 32
    """Output cap for generative judges. Ignored by judges that emit a numeric score directly."""

    prompt: str | None = None
    """Optional override prompt text. ``None`` means the plugin uses its default."""

    prompt_variant: str = "lenient_binary"
    """Name of the prompt variant (e.g. ``youcook2_binary``). Passed to plugins that
    maintain their own prompt map (e.g. VCI) so they can select the right template."""

    extra: dict[str, Any] = attrs.Factory(dict)
    """Plugin-specific knobs (e.g. retry counts, API model name)."""


@attrs.define
class JudgeItem:
    """A single (caption, GT) pair plus optional evidence."""

    caption: str
    gt_action_text: str
    """Ground-truth action description, possibly empty for reference-free judges."""

    # Optional evidence — at most one is populated, depending on the plugin's evidence_kind.
    frames: "torch.Tensor | None" = None
    """Decoded frames as tensor of shape (T, C, H, W) in [0, 1]."""

    mp4_bytes: bytes | None = None
    """Raw MP4 bytes for plugins that send the encoded clip directly to an API."""

    metadata: dict[str, Any] = attrs.Factory(dict)
    """Free-form context (source_video, window_key, gt_skill, ...). Returned in JudgeResult."""


@attrs.define
class JudgeResult:
    """Outcome of judging one ``JudgeItem``.

    ``verdict`` is plugin-defined but the convention is::

        "correct" | "incorrect" | "incorrect_object" | "incorrect_action"

    ``score`` is normalised to a 1-5 integer to remain compatible with the existing
    ``check_scores.py`` tooling (5 = correct, anything < 5 = flagged).
    """

    verdict: str | None
    score: int
    explanation: str
    raw_output: str = ""
    metadata: dict[str, Any] = attrs.Factory(dict)


class JudgePlugin(ABC):
    """Judge plugin interface.

    Implementations are stateful: ``setup()`` is called once per worker, and
    ``judge_batch()`` is then called repeatedly with batches of items.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @staticmethod
    @abstractmethod
    def variant() -> str:
        """Return the unique judge identifier (e.g. ``gemma4_e4b``)."""

    @classmethod
    def model_id(cls) -> str | None:
        """Return the HuggingFace model ID for local-model plugins.

        Returns ``None`` for API-only judges. Default implementation looks the variant up
        in the registry; override if a plugin doesn't have a HF id.
        """
        return get_judge_model_id(cls.variant())

    @classmethod
    def model_path(cls) -> Path | None:
        """Return the local path to the model weights, if applicable."""
        model_id = cls.model_id()
        if model_id is None:
            return None
        return model_utils.get_local_dir_for_weights_name(model_id)

    # ── Capability declarations ───────────────────────────────────────────────

    @property
    @abstractmethod
    def evidence_kind(self) -> EvidenceKind:
        """What evidence the plugin needs alongside the caption."""

    @property
    @abstractmethod
    def conda_env_name(self) -> str | None:
        """Pixi environment to run this judge in (e.g. ``unified``).

        Return ``None`` to use the pipeline's default env.
        """

    @property
    @abstractmethod
    def resources(self) -> CuratorStageResource:
        """Resource requirements for one worker (CPUs / GPUs)."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def setup(self, config: JudgeConfig) -> None:
        """Called once per worker before any ``judge_batch`` call.

        Plugins should load their model / build their API client here. The ``config``
        is the same instance that will be visible to subsequent ``judge_batch`` calls
        through ``self``, so plugins typically store it.
        """

    @abstractmethod
    def judge_batch(self, items: list[JudgeItem]) -> list[JudgeResult]:
        """Judge a batch of items.

        Returned list MUST be the same length and order as ``items``. If a single item
        fails, return a ``JudgeResult`` with ``verdict=None`` for that index rather than
        raising; the framework tracks failures per-item.
        """
