# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loaders for the cosmos-curate output layout.

The split-annotate pipeline writes (see ``ClipWriterStage``):

- ``metas/v0/{clip_uuid}.json``          per-clip metadata (span, source video, windows)
- ``v0/all_window_captions.json``        {video: {clip_uuid: {"{start}_{end}": caption}}}
- ``v0/all_window_judgments.json``       {video: {clip_uuid: {"{start}_{end}": {variant: {...}}}}}
- ``{algo}_embd/{clip_uuid}.pickle``     per-clip embedding (pickled numpy array)
- ``{algo}_embd_parquet/*.parquet``      grouped embeddings with columns ``id``, ``embedding``

These loaders read that layout directly so an evaluation never needs to import the
main package.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from downstream_eval.common.types import ClipRecord, Segment, WindowRecord

_EMBED_DIR_BY_ALGO = {
    "internvideo2": "iv2_embd",
    "openai": "openai_embd",
    "cradio": "cradio_embd",
}


def _embed_dirname(algorithm: str) -> str:
    """Map an embedding algorithm name to its output sub-directory."""
    if algorithm.startswith("cosmos-embed1"):
        return "ce1_embd"
    return _EMBED_DIR_BY_ALGO.get(algorithm, f"{algorithm}_embd")


def _parse_window(raw: dict[str, Any]) -> WindowRecord:
    """Parse a single window dict from a per-clip metadata file."""
    captions = {
        key[: -len("_caption")]: str(value)
        for key, value in raw.items()
        if key.endswith("_caption") and not key.endswith("_enhanced_caption")
    }
    judge = raw.get("judge", {})
    return WindowRecord(
        start_frame=int(raw.get("start_frame", 0)),
        end_frame=int(raw.get("end_frame", 0)),
        captions=captions,
        judge=judge if isinstance(judge, dict) else {},
    )


def load_clip_records(output_dir: str | Path, version: str = "v0") -> list[ClipRecord]:
    """Load per-clip metadata records from ``metas/{version}/*.json``.

    Args:
        output_dir: Pipeline ``--output-clip-path`` root.
        version: Metadata version sub-directory (default ``v0``).

    Returns:
        One :class:`ClipRecord` per clip metadata file, sorted by source video then start time.

    """
    metas_dir = Path(output_dir) / "metas" / version
    records: list[ClipRecord] = []
    for path in sorted(metas_dir.glob("*.json")):
        data = json.loads(path.read_text())
        span = data.get("duration_span") or [0.0, 0.0]
        windows = [_parse_window(w) for w in data.get("windows", [])]
        records.append(
            ClipRecord(
                uuid=str(data.get("span_uuid", path.stem)),
                source_video=str(data.get("source_video", "")),
                span=(float(span[0]), float(span[1])),
                windows=windows,
                framerate=data.get("framerate_source") or data.get("framerate"),
            )
        )
    records.sort(key=lambda r: (r.source_video, r.span[0]))
    return records


def load_window_captions(output_dir: str | Path, version: str = "v0") -> dict[str, Any]:
    """Load ``{version}/all_window_captions.json`` (empty dict if absent)."""
    path = Path(output_dir) / version / "all_window_captions.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def load_window_judgments(output_dir: str | Path, version: str = "v0") -> dict[str, Any]:
    """Load ``{version}/all_window_judgments.json`` (empty dict if absent)."""
    path = Path(output_dir) / version / "all_window_judgments.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def load_clip_embeddings(output_dir: str | Path, algorithm: str = "cradio") -> dict[str, npt.NDArray[np.float32]]:
    """Load per-clip embeddings from ``{algo}_embd/*.pickle``.

    Falls back to the grouped parquet files when no pickles are present and a
    parquet engine is installed.

    Args:
        output_dir: Pipeline output root.
        algorithm: Embedding algorithm name (``cradio``, ``internvideo2``, ...).

    Returns:
        Mapping of clip-uuid -> 1-D embedding vector.

    """
    out: dict[str, npt.NDArray[np.float32]] = {}
    embd_dir = Path(output_dir) / _embed_dirname(algorithm)
    if embd_dir.is_dir():
        for path in sorted(embd_dir.glob("*.pickle")):
            with path.open("rb") as fh:
                arr = pickle.load(fh)  # noqa: S301 - trusted local pipeline output
            out[path.stem] = np.asarray(arr, dtype=np.float32).reshape(-1)
    if not out:
        out.update(_load_embeddings_from_parquet(output_dir, algorithm))
    return out


def _load_embeddings_from_parquet(output_dir: str | Path, algorithm: str) -> dict[str, npt.NDArray[np.float32]]:
    """Best-effort parquet embedding loader; returns empty dict if no engine is available."""
    parquet_dir = Path(output_dir) / f"{_embed_dirname(algorithm)}_parquet"
    if not parquet_dir.is_dir():
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}
    out: dict[str, npt.NDArray[np.float32]] = {}
    for path in sorted(parquet_dir.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except (ImportError, ValueError, OSError):
            return out
        for row in frame.itertuples(index=False):
            out[str(row.id)] = np.asarray(row.embedding, dtype=np.float32).reshape(-1)
    return out


def predicted_segments_by_video(clips: list[ClipRecord]) -> dict[str, list[Segment]]:
    """Group clip time spans into ordered predicted segments per source video."""
    by_video: dict[str, list[Segment]] = defaultdict(list)
    for clip in clips:
        by_video[clip.source_video].append(clip.to_segment())
    for segments in by_video.values():
        segments.sort(key=lambda s: s.start)
    return dict(by_video)
