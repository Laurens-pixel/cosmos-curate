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
"""Cosmos-Curate judge plugin interface.

Mirrors the design of ``vllm_interface.py`` but for caption-judge models. The
``JudgeStage`` does not import individual plugin classes — it goes through this
registry, so adding a new judge is a single-file change plus one registry line.

Usage from a stage::

    plugin = get_judge_plugin(variant)
    plugin.setup(config)
    results = plugin.judge_batch(items)

"""

from typing import cast

from cosmos_curate.models.judge_gemma4 import JudgeGemma4_31B, JudgeGemma4E4B
from cosmos_curate.models.judge_gemma4_video import JudgeGemma4_31BVideo, JudgeGemma4E4BVideo
from cosmos_curate.models.judge_plugin import JudgePlugin
from cosmos_curate.models.judge_qwen3vl import JudgeQwen3VL30B, JudgeQwen3VL30BFP8
from cosmos_curate.models.judge_vci import JudgeVCI3B, JudgeVCI7B

# Add new judge plugins here.
_JUDGE_PLUGINS: dict[str, type[JudgePlugin]] = {
    JudgeGemma4E4B.variant(): JudgeGemma4E4B,
    JudgeGemma4_31B.variant(): JudgeGemma4_31B,
    JudgeGemma4E4BVideo.variant(): JudgeGemma4E4BVideo,
    JudgeGemma4_31BVideo.variant(): JudgeGemma4_31BVideo,
    JudgeVCI3B.variant(): JudgeVCI3B,
    JudgeVCI7B.variant(): JudgeVCI7B,
    JudgeQwen3VL30B.variant(): JudgeQwen3VL30B,
    JudgeQwen3VL30BFP8.variant(): JudgeQwen3VL30BFP8,
}


def list_judge_variants() -> list[str]:
    """Return the sorted list of registered judge variants."""
    return sorted(_JUDGE_PLUGINS.keys())


def get_judge_plugin_class(variant: str) -> type[JudgePlugin]:
    """Return the plugin *class* for ``variant``.

    Useful when you need static metadata (e.g. ``conda_env_name``, ``resources``) without
    constructing the plugin yet.
    """
    if variant not in _JUDGE_PLUGINS:
        msg = f"Unknown judge variant: {variant!r}. Registered variants: {list_judge_variants()}"
        raise ValueError(msg)
    return cast("type[JudgePlugin]", _JUDGE_PLUGINS[variant])


def make_judge_plugin(variant: str) -> JudgePlugin:
    """Instantiate the judge plugin for ``variant``."""
    cls = get_judge_plugin_class(variant)
    return cls()
