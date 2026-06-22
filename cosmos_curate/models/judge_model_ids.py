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
"""Variant -> HuggingFace model ID mapping for judge plugins.

API-only judges (e.g. Gemini, OpenAI) have ``None`` here.
"""

_JUDGE_MODELS: dict[str, str | None] = {
    "gemma4_e4b": "google/gemma-4-E4B-it",
    "gemma4_31b": "google/gemma-4-31B-it",
    "gemma4_e4b_video": "google/gemma-4-E4B-it",
    "gemma4_31b_video": "google/gemma-4-31B-it",
    "vci_3b": "dipta007/VCInspector-3B",
    "vci_7b": "dipta007/VCInspector-7B",
    "qwen3vl_30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3vl_30b_fp8": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
}


def get_judge_model_id(variant: str) -> str | None:
    """Return the HuggingFace model ID for the variant, or ``None`` for API-only judges.

    Raises ``KeyError`` if the variant is unknown so callers can distinguish unknown
    variants from API-only judges.
    """
    if variant not in _JUDGE_MODELS:
        msg = f"Unknown judge variant: {variant!r}. Register it in judge_model_ids.py."
        raise KeyError(msg)
    return _JUDGE_MODELS[variant]
