# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-driven model/judge parameter registry.

Loads ``configs/model_registry.yaml`` (and an optional user override pointed to by the
``COSMOS_MODEL_REGISTRY`` env var) and exposes per-variant parameter dicts for caption
and judge models.

This layer is *additive*: plugins that opt in read their knobs here in ``setup()``;
plugins that don't keep using their in-file constants. Nothing breaks if the YAML is
missing — callers always pass a ``default`` and get it back.

Example::

    from cosmos_curate.models import model_registry

    params = model_registry.get_judge_params("qwen3vl_30b")
    fps = params.get("sampling_fps", 2.0)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

_DEFAULT_REGISTRY = Path(__file__).parent / "configs" / "model_registry.yaml"
_OVERRIDE_ENV = "COSMOS_MODEL_REGISTRY"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base`` one level deep per top-level section.

    Within ``judges`` / ``captions`` each variant entry from ``override`` fully
    replaces the matching base entry (shallow per-variant), which is the least
    surprising behaviour for a config override.
    """
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for section, entries in override.items():
        if isinstance(entries, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **entries}
        else:
            merged[section] = entries
    return merged


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    """Load and cache the merged registry (defaults + optional user override)."""
    data: dict[str, Any] = {}
    if _DEFAULT_REGISTRY.exists():
        try:
            data = yaml.safe_load(_DEFAULT_REGISTRY.read_text()) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - config error path
            logger.warning(f"model_registry: failed to parse {_DEFAULT_REGISTRY}: {exc}")

    override_path = os.environ.get(_OVERRIDE_ENV)
    if override_path:
        p = Path(override_path)
        if p.exists():
            try:
                override = yaml.safe_load(p.read_text()) or {}
                data = _deep_merge(data, override)
                logger.info(f"model_registry: merged override from {p}")
            except yaml.YAMLError as exc:  # pragma: no cover - config error path
                logger.warning(f"model_registry: failed to parse override {p}: {exc}")
        else:
            logger.warning(f"model_registry: {_OVERRIDE_ENV}={p} does not exist; ignoring")
    return data


def get_judge_params(variant: str) -> dict[str, Any]:
    """Return the parameter dict for a judge ``variant`` (empty dict if absent)."""
    return dict(_load_registry().get("judges", {}).get(variant, {}))


def get_caption_params(variant: str) -> dict[str, Any]:
    """Return the parameter dict for a caption ``variant`` (empty dict if absent)."""
    return dict(_load_registry().get("captions", {}).get(variant, {}))


def list_judges() -> list[str]:
    """Return all judge variants declared in the registry."""
    return sorted(_load_registry().get("judges", {}).keys())


def list_captions() -> list[str]:
    """Return all caption variants declared in the registry."""
    return sorted(_load_registry().get("captions", {}).keys())
