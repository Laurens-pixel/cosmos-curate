# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth source plugins for the evaluation phase."""

from cosmos_curate.pipelines.video.evaluation.gt_sources.agibot import AgibotTaskInfoGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.assembly101 import Assembly101Gt
from cosmos_curate.pipelines.video.evaluation.gt_sources.gt_source import GtSource
from cosmos_curate.pipelines.video.evaluation.gt_sources.inhard import InHardGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.inhard_online import InHardOnlineGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.manual import ManualAnnotationJsonGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.none import NoneGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.nuscenes import NuScenesGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.unitree_g1 import UnitreeG1Gt
from cosmos_curate.pipelines.video.evaluation.gt_sources.wearable_ai import WearableAiGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.wgo import WgoBenchGt
from cosmos_curate.pipelines.video.evaluation.gt_sources.youcook2 import YouCook2Gt

_GT_SOURCES: dict[str, type[GtSource]] = {
    AgibotTaskInfoGt.name(): AgibotTaskInfoGt,
    Assembly101Gt.name(): Assembly101Gt,
    InHardGt.name(): InHardGt,
    InHardOnlineGt.name(): InHardOnlineGt,
    ManualAnnotationJsonGt.name(): ManualAnnotationJsonGt,
    NoneGt.name(): NoneGt,
    NuScenesGt.name(): NuScenesGt,
    UnitreeG1Gt.name(): UnitreeG1Gt,
    WearableAiGt.name(): WearableAiGt,
    WgoBenchGt.name(): WgoBenchGt,
    YouCook2Gt.name(): YouCook2Gt,
}


def list_gt_sources() -> list[str]:
    """Return registered GT source names."""
    return sorted(_GT_SOURCES.keys())


def get_gt_source_class(name: str) -> type[GtSource]:
    """Return the GT source class registered under ``name``."""
    if name not in _GT_SOURCES:
        msg = f"Unknown GT source: {name!r}. Registered: {list_gt_sources()}"
        raise ValueError(msg)
    return _GT_SOURCES[name]


def make_gt_source(name: str, cfg: dict) -> GtSource:
    """Instantiate the GT source registered under ``name`` from a dataset config dict."""
    return get_gt_source_class(name).from_config(cfg)


__all__ = [
    "AgibotTaskInfoGt",
    "Assembly101Gt",
    "GtSource",
    "InHardGt",
    "InHardOnlineGt",
    "ManualAnnotationJsonGt",
    "NoneGt",
    "NuScenesGt",
    "UnitreeG1Gt",
    "WearableAiGt",
    "WgoBenchGt",
    "YouCook2Gt",
    "get_gt_source_class",
    "list_gt_sources",
    "make_gt_source",
]
