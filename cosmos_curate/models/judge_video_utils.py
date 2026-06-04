# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared video-frame decoding utilities for judge plugins.

VCInspector (judge_vci.py) and Gemma4-Video (judge_gemma4_video.py) both decode MP4
bytes into PIL images before inference. This module provides one implementation so
frame-sampling parameters stay consistent across plugins.
"""

import io

from PIL import Image

# VCInspector paper defaults — used as the base reference.
DEFAULT_SAMPLING_FPS: float = 1.0
DEFAULT_MAX_FRAMES: int = 32
DEFAULT_FRAME_SIZE: int = 224


def decode_frames(
    mp4_bytes: bytes,
    *,
    sampling_fps: float = DEFAULT_SAMPLING_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_size: int = DEFAULT_FRAME_SIZE,
) -> list[Image.Image]:
    """Decode up to *max_frames* PIL images from raw MP4 bytes.

    The sampling interval adapts to clip length: for clips longer than
    ``max_frames / sampling_fps`` seconds the interval is stretched so all
    frames are spread across the whole clip.  Short clips are sampled at
    *sampling_fps* and may return fewer than *max_frames* frames.
    """
    import av  # available in unified pixi env

    container = av.open(io.BytesIO(mp4_bytes))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    duration_s = float(stream.duration * stream.time_base) if stream.duration else None
    base_interval = 1.0 / sampling_fps
    if duration_s and duration_s > 0:
        interval = max(base_interval, duration_s / max_frames)
    else:
        interval = base_interval

    frames: list[Image.Image] = []
    next_s = 0.0
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        if t >= next_s:
            img = frame.to_image().resize((frame_size, frame_size), Image.LANCZOS)
            frames.append(img)
            next_s += interval
            if len(frames) >= max_frames:
                break
        if duration_s is not None and t > duration_s:
            break
    container.close()
    return frames
