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
"""Pluggable shot/action boundary detectors (Class A/B/C/D)."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw, ImageFont
from loguru import logger  # type: ignore[import-not-found]

def _resolve_pretrained_path(model_id: str, *, fallbacks: tuple[str, ...] = ()) -> str:
    """Resolve a HuggingFace model id to a local path when weights are bind-mounted."""
    for candidate_id in (model_id, *fallbacks):
        parts = candidate_id.split("/", 1)
        if len(parts) != 2:
            continue
        org, name = parts
        for base in (Path("/config/models"), Path(os.environ.get("HF_HOME", ""))):
            if not str(base):
                continue
            candidate = base / org / name
            if (candidate / "config.json").exists():
                return str(candidate)
    return model_id


def _decode_frames_pyav(
    video_path: str,
    target_times_s: list[float],
) -> tuple[list[npt.NDArray[np.uint8]], list[float]]:
    """Decode RGB frames at the given target times with PyAV — the AV1-safe frame reader.

    ``cv2.VideoCapture`` silently returns no frames on AV1-encoded video (WGO, AgiBot), so every
    frame-based shot-boundary detector saw an empty video and emitted zero boundaries. PyAV
    decodes AV1 correctly. Frames are read sequentially and the first frame at/after each target
    time is kept, so memory stays bounded to the number of sampled frames (not the whole video).
    """
    import av  # noqa: PLC0415 — heavy optional dep, imported on use

    targets = sorted(t for t in target_times_s if t >= 0)
    if not targets:
        return [], []
    try:
        container = av.open(video_path)
    except (av.FFmpegError, OSError):
        return [], []
    frames: list[npt.NDArray[np.uint8]] = []
    kept: list[float] = []
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        # Seek to the keyframe at/before the first target instead of always decoding from frame
        # 0 — TPIVOT calls this many times per video (per refinement iteration, per seed), so
        # without seeking, every call re-decodes the whole video prefix leading up to its window.
        if stream.time_base:
            try:
                container.seek(int(targets[0] / stream.time_base), stream=stream, backward=True, any_frame=False)
            except av.FFmpegError:
                pass
        ti = 0
        for frame in container.decode(stream):
            if ti >= len(targets):
                break
            t = float(frame.pts * stream.time_base) if (frame.pts is not None and stream.time_base) else 0.0
            while ti < len(targets) and t >= targets[ti]:
                frames.append(np.asarray(frame.to_rgb().to_image(), dtype=np.uint8))
                kept.append(targets[ti])
                ti += 1
    finally:
        container.close()
    return frames, kept


def _extract_frames_uniform(
    video_path: str,
    fps: float,
    *,
    max_frames: int | None = None,
) -> tuple[list[npt.NDArray[np.uint8]], list[float]]:
    duration = _video_duration_s(video_path)
    if duration <= 0:
        return [], []
    step = 1.0 / max(fps, 1e-3)
    targets = [i * step for i in range(int(duration / step) + 1)]
    if max_frames is not None:
        targets = targets[:max_frames]
    return _decode_frames_pyav(video_path, targets)


def _extract_frames_window(
    video_path: str,
    center_s: float,
    window_s: float,
    n_frames: int,
) -> tuple[list[npt.NDArray[np.uint8]], list[float]]:
    duration = _video_duration_s(video_path)
    t0 = max(0.0, center_s - window_s / 2.0)
    t1 = min(duration, center_s + window_s / 2.0)
    if t1 <= t0:
        t1 = min(duration, t0 + 1.0)
    targets = np.linspace(t0, t1, n_frames).tolist()
    return _decode_frames_pyav(video_path, targets)


def _moving_average(features: npt.NDArray[np.float32], kernel: int) -> npt.NDArray[np.float32]:
    if kernel <= 1:
        return features
    pad = kernel // 2
    feat = np.pad(features, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(features, dtype=np.float32)
    for i in range(features.shape[0]):
        out[i] = feat[i : i + kernel].mean(axis=0)
    return out


def _adjacent_cosine(features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    f = features / norms
    return np.sum(f[:-1] * f[1:], axis=1)


def _abd_boundaries(
    features: npt.NDArray[np.float32],
    timestamps: list[float],
    *,
    alpha: float,
) -> list[float]:
    """Detect boundaries from embedding dips without a fixed segment count."""
    n = features.shape[0]
    if len(timestamps) < 3 or n < 3:
        return []
    # Smoothing kernel scales with sequence length only (no target segment count).
    kernel = max(int(alpha * n / 20), 1)
    smoothed = _moving_average(features, kernel)
    sim = _adjacent_cosine(smoothed)
    if sim.shape[0] < 3:
        return []
    cands: list[int] = []
    for i in range(1, sim.shape[0] - 1):
        if sim[i] <= sim[i - 1] and sim[i] <= sim[i + 1]:
            cands.append(i)
    if not cands:
        return []
    sim_mean = float(sim.mean())
    sim_std = float(sim.std()) if float(sim.std()) > 1e-8 else 0.0
    cutoff = sim_mean - alpha * sim_std
    keep = [i for i in cands if sim[i] <= cutoff]
    if not keep:
        keep = [min(cands, key=lambda i: sim[i])]
    keep_sorted = sorted(set(keep))
    return [timestamps[min(i, len(timestamps) - 1)] for i in keep_sorted]


def _label_font(size: int) -> tuple[ImageFont.ImageFont, int]:
    """Load the largest legible label font available, with its effective point size.

    VLMs downsample large contact-sheet/grid images internally before inference, so a label
    that's only readable at full tile resolution — like PIL's tiny fixed-size default bitmap
    font — becomes illegible noise by the time the model sees it: it was never actually able
    to read the frame index / timestamp it was being asked to reason about.
    """
    try:
        return ImageFont.load_default(size=size), size
    except TypeError:  # Pillow < 10.1 has no `size` kwarg for load_default()
        return ImageFont.load_default(), 11


def _compose_grid(
    frames: list[npt.NDArray[np.uint8]],
    *,
    grid_cols: int,
    frame_size: tuple[int, int],
) -> Image.Image:
    n = len(frames)
    rows = math.ceil(n / grid_cols)
    fw, fh = frame_size
    pad = 4
    canvas = Image.new("RGB", (grid_cols * (fw + pad) + pad, rows * (fh + pad) + pad), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font, font_size = _label_font(max(28, fh // 6))
    box_w, box_h = font_size * len(str(n)) // 2 + 16, font_size + 14
    for i, fr in enumerate(frames):
        r, c = i // grid_cols, i % grid_cols
        x = pad + c * (fw + pad)
        y = pad + r * (fh + pad)
        img = Image.fromarray(fr).resize((fw, fh), Image.Resampling.BICUBIC)
        canvas.paste(img, (x, y))
        label = str(i + 1)
        draw.rectangle([x + fw - box_w, y, x + fw, y + box_h], fill=(0, 0, 0))
        draw.text((x + fw - box_w + 8, y + 5), label, fill=(255, 255, 0), font=font)
    return canvas


def _parse_json_frame_indices(text: str, *, n_frames: int) -> list[int]:
    cleaned = _strip_think(text)

    def _ints_in_range(values: Any) -> list[int]:  # noqa: ANN401
        vals = values if isinstance(values, list) else [values]
        out: list[int] = []
        for x in vals:
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= v <= n_frames:
                out.append(v)
        return out

    for obj in _iter_json_objects(cleaned):
        for key in ("frame_index", "boundaries", "transition_frames", "transitions", "indices"):
            if key in obj:
                got = _ints_in_range(obj[key])
                if got:
                    return got
    # Fallback: any integers in range (handles models that answer in prose, not JSON).
    return _ints_in_range(re.findall(r"\b(\d+)\b", cleaned))


def _video_duration_s(video_path: str) -> float:
    import av  # noqa: PLC0415

    try:
        container = av.open(video_path)
    except (av.FFmpegError, OSError):
        return 0.0
    try:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
    finally:
        container.close()
    return 0.0


def _extract_frames_at_interval(
    video_path: str,
    interval_s: float,
) -> tuple[list[npt.NDArray[np.uint8]], list[float]]:
    duration = _video_duration_s(video_path)
    if duration <= 0:
        return [], []
    step = max(interval_s, 1e-3)
    n = int(duration / step) + 1
    targets = [i * step for i in range(n)]
    return _decode_frames_pyav(video_path, targets)


def _burn_timestamp_tile(
    frame: npt.NDArray[np.uint8],
    timestamp_s: float,
    tile_size: int,
) -> Image.Image:
    img = Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    font, font_size = _label_font(max(28, tile_size // 6))
    label = f"{timestamp_s:.1f}s"
    box_w, box_h = font_size * len(label) // 2 + 16, font_size + 14
    draw.rectangle([2, 2, 2 + box_w, 2 + box_h], fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 0), font=font)
    return img


def _compose_macrodata_sheet(
    tiles: list[Image.Image],
    *,
    sheet_columns: int,
    sheet_rows: int,
) -> Image.Image:
    pad = 4
    fw = tiles[0].width if tiles else 224
    fh = tiles[0].height if tiles else 224
    canvas = Image.new(
        "RGB",
        (sheet_columns * (fw + pad) + pad, sheet_rows * (fh + pad) + pad),
        (20, 20, 20),
    )
    for i, tile in enumerate(tiles):
        r, c = i // sheet_columns, i % sheet_columns
        x = pad + c * (fw + pad)
        y = pad + r * (fh + pad)
        canvas.paste(tile, (x, y))
    return canvas


def _macrodata_prompt(
    *,
    duration_s: float,
    sample_interval_sec: float,
    n_sheets: int,
    dur_min: float,
    dur_max: float,
    retry_error: str | None = None,
    attempt: int = 0,
) -> str:
    prompt = (
        "You are segmenting a video of a manipulation task into its individual subtask steps. "
        "The actor performing the task may be a human hand/arm or a robot gripper/arm.\n\n"
        f"The {n_sheets} attached contact sheet(s) show chronologically ordered frames "
        f"sampled every {sample_interval_sec:.1f}s across the full episode.\n"
        "Each tile has its timestamp burned in at the top-left corner (e.g. \"12.5s\").\n"
        f"Episode duration: {duration_s:.2f}s.\n\n"
        "Segment into EVERY distinct step, in temporal order:\n"
        "- Each grasp/pick, each place/release, and each open, close, pour, push or wipe of an "
        "object is its OWN segment. A pick-and-place is TWO segments: the pick, then the place.\n"
        "- Start a new segment whenever the actor begins acting on a different object, switches "
        "between picking and placing, or begins a new distinct motion.\n"
        "- Do NOT collapse the episode into one segment — most episodes have many "
        "short steps; list all of them.\n"
        f"- Segments are typically {dur_min:.0f}-{dur_max:.0f}s. Use the burned-in tile "
        "timestamps for start_sec/end_sec; segments should tile the episode in order with no gaps.\n"
        "- Return only the JSON object, nothing else.\n\n"
        "Example (an episode that picks and places two objects — note four segments):\n"
        '{"segments": ['
        '{"start_sec": 0.0, "end_sec": 3.5, "subtask": "pick up the red block"}, '
        '{"start_sec": 3.5, "end_sec": 6.0, "subtask": "place the red block in the bin"}, '
        '{"start_sec": 6.0, "end_sec": 9.0, "subtask": "pick up the blue cup"}, '
        '{"start_sec": 9.0, "end_sec": 12.0, "subtask": "place the blue cup on the shelf"}]}'
    )
    if retry_error:
        if "empty" in retry_error.lower():
            # {"segments": []} is valid JSON, so a generic "return valid JSON" nudge does nothing —
            # the model just repeats the empty list. Force it to actually segment. The demanded
            # minimum grows with each attempt so a GREEDY model (deterministic on a fixed prompt)
            # gets a different, stronger instruction each retry instead of repeating itself.
            min_segs = 2 + attempt
            prompt += (
                f"\n\nYour previous answer had ZERO segments, which is INVALID. This episode DOES "
                f"contain manipulation activity across its full {duration_s:.0f}s duration. Look again "
                f"at the contact sheets and return a NON-EMPTY list of AT LEAST {min_segs} segments "
                "covering the distinct action phases in temporal order (each pick/place/manipulate "
                "step). Return only the JSON object."
            )
        else:
            prompt += f"\n\nPrevious response failed validation: {retry_error}\nPlease fix and return valid JSON only."
    return prompt


@dataclass
class MacrodataSegment:
    start_sec: float
    end_sec: float
    subtask: str


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning and code fences before JSON extraction.

    Reasoning VLMs (e.g. GLM-4.1V-Thinking) emit a long <think> block — often containing braces —
    before the JSON answer, and may be truncated mid-think when the token budget runs out. Both
    corrupt a greedy brace match, so strip closed and trailing-unclosed think blocks and fences.
    """
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", " ", text, flags=re.DOTALL | re.IGNORECASE)  # truncated / unclosed
    return re.sub(r"```(?:json)?|```", " ", text)


def _iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yield each balanced ``{...}`` substring that parses as a JSON object, last-first.

    A balanced-brace scan rather than a greedy regex, so stray braces in prose or reasoning
    before the real answer cannot corrupt extraction. Last-first because reasoning models emit
    reasoning-then-answer, so the final object is the intended output.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
    for span in reversed(spans):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _parse_macrodata_response(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    cleaned = _strip_think(text)
    # 1. Normal case: a complete {"segments": [...]} object.
    for obj in _iter_json_objects(cleaned):
        segments = obj.get("segments")
        if isinstance(segments, list):
            return segments, None
    # 2. Truncated output (VLM hit the token limit mid-array): the outer object never closes, so
    #    _iter_json_objects finds nothing. Recover every COMPLETE inner segment object instead —
    #    an answer cut off at segment 18 still yields the first 17 usable segments.
    recovered: list[dict[str, Any]] = []
    for m in re.finditer(r"\{[^{}]*?\"start_sec\"[^{}]*?\}", cleaned, flags=re.DOTALL):
        try:
            seg = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(seg, dict) and "start_sec" in seg:
            recovered.append(seg)
    if recovered:
        return recovered, None
    return None, "No JSON object with a 'segments' list found in model output"


def _validate_macrodata_segments(
    raw_segments: list[dict[str, Any]],
    duration_s: float,
) -> tuple[list[MacrodataSegment], str | None]:
    if not raw_segments:
        return [], "segments list is empty"
    # Per-segment tolerant validation: clamp out-of-range timestamps and skip individual bad
    # segments, keeping the rest. The old all-or-nothing version discarded a whole 18-segment
    # answer because one segment overshot the duration by 0.001s (a rounding artifact at the
    # video end) — throwing away 17 good segments.
    parsed: list[MacrodataSegment] = []
    prev_start = -1.0
    for seg in raw_segments:
        if not isinstance(seg, dict) or "start_sec" not in seg or "end_sec" not in seg or "subtask" not in seg:
            continue
        try:
            start = float(seg["start_sec"])
            end = float(seg["end_sec"])
        except (TypeError, ValueError):
            continue
        subtask = str(seg["subtask"]).strip()
        if not subtask:
            continue
        # clamp to episode bounds rather than rejecting (VLMs overshoot the final timestamp)
        start = min(max(start, 0.0), duration_s)
        end = min(max(end, 0.0), duration_s)
        if end <= start or start < prev_start - 1e-6:
            continue  # skip degenerate or non-monotonic, keep the rest
        prev_start = start
        parsed.append(MacrodataSegment(start_sec=start, end_sec=end, subtask=subtask))
    if not parsed:
        return [], "no valid segments after clamping/filtering"
    return parsed, None


class BaseVLMClient(Protocol):
    def query(self, image: Image.Image, text_prompt: str, max_new_tokens: int = 256) -> str: ...

    def query_multi(self, images: list[Image.Image], text_prompt: str, max_new_tokens: int = 256) -> str: ...


class Glm4vVLMClient:
    """GLM-V backend for prompt-based boundary localization.

    ``model_id`` selects the checkpoint. GLM-4.1V-9B-Thinking (default) and GLM-4.6V-Flash (9B,
    newer generation) share the ``Glm4vForConditionalGeneration`` architecture, so a newer
    checkpoint is a drop-in — only the weights change. The 106B GLM-4.5V/4.6V are deliberately not
    offered: at ~212 GB they would need every GPU in the job, leaving none for the captioning and
    judging stages that run alongside the segmenter.
    """

    MODEL_ID = "zai-org/GLM-4.1V-9B-Thinking"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or self.MODEL_ID
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor

        model_path = _resolve_pretrained_path(self.model_id)
        local_only = model_path != self.model_id
        if model_path == self.model_id and not Path(model_path).exists():
            msg = (
                f"GLM-V weights not found locally for {self.model_id}. "
                f"Download to /config/models/{self.model_id}/ or set HF_HOME."
            )
            raise RuntimeError(msg)
        try:
            from transformers import Glm4vForConditionalGeneration as model_cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as model_cls

        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model = model_cls.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=local_only,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)
        self._model.eval()

    def query(self, image: Image.Image, text_prompt: str, max_new_tokens: int = 256) -> str:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def query_multi(self, images: list[Image.Image], text_prompt: str, max_new_tokens: int = 256) -> str:
        if len(images) == 1:
            return self.query(images[0], text_prompt, max_new_tokens=max_new_tokens)
        self._load()
        assert self._processor is not None
        assert self._model is not None
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class Qwen3VLMClient:
    """Qwen3-VL-8B backend for prompt-based boundary localization."""

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_path = _resolve_pretrained_path(self.MODEL_ID)
        local_only = model_path != self.MODEL_ID
        if model_path == self.MODEL_ID and not Path(model_path).exists():
            msg = (
                f"Qwen3-VL weights not found locally for {self.MODEL_ID}. "
                "Download to /config/models/Qwen/Qwen3-VL-8B-Instruct/ or set HF_HOME."
            )
            raise RuntimeError(msg)
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=local_only,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)
        self._model.eval()

    def query(self, image: Image.Image, text_prompt: str, max_new_tokens: int = 256) -> str:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def query_multi(self, images: list[Image.Image], text_prompt: str, max_new_tokens: int = 256) -> str:
        if len(images) == 1:
            return self.query(images[0], text_prompt, max_new_tokens=max_new_tokens)
        self._load()
        assert self._processor is not None
        assert self._model is not None
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class Molmo2VLMClient:
    """Molmo2-8B backend for prompt-based boundary localization."""

    MODEL_ID = "allenai/Molmo2-8B"

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_path = _resolve_pretrained_path(self.MODEL_ID)
        local_only = model_path != self.MODEL_ID
        if model_path == self.MODEL_ID and not Path(model_path).exists():
            msg = (
                f"Molmo2 weights not found locally for {self.MODEL_ID}. "
                "Download to /config/models/allenai/Molmo2-8B/ or set HF_HOME."
            )
            raise RuntimeError(msg)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=local_only,
        ).to(self._device)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        self._model.eval()

    def query(self, image: Image.Image, text_prompt: str, max_new_tokens: int = 256) -> str:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.tokenizer.decode(trimmed[0], skip_special_tokens=True).strip()

    def query_multi(self, images: list[Image.Image], text_prompt: str, max_new_tokens: int = 256) -> str:
        if len(images) == 1:
            return self.query(images[0], text_prompt, max_new_tokens=max_new_tokens)
        self._load()
        assert self._processor is not None
        assert self._model is not None
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.tokenizer.decode(trimmed[0], skip_special_tokens=True).strip()


class InternVL3Client:
    """InternVL3 backend for prompt-based boundary localization."""

    def __init__(self, model_size: str = "8B") -> None:
        size_map = {
            "2B": "OpenGVLab/InternVL3-2B",
            "8B": "OpenGVLab/InternVL3-8B",
            "14B": "OpenGVLab/InternVL3-14B",
        }
        self._model_id = size_map.get(model_size, "OpenGVLab/InternVL3-8B")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        model_path = _resolve_pretrained_path(self._model_id)
        if model_path == self._model_id and not Path(model_path).exists():
            msg = (
                f"InternVL3 weights not found locally for {self._model_id}. "
                "Download to /config/models/OpenGVLab/<model-name>/ or set HF_HOME."
            )
            raise RuntimeError(msg)
        self._model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            trust_remote_code=True,
            local_files_only=model_path != self._model_id,
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=model_path != self._model_id,
        )
        self._model.eval()

    def query(self, image: Image.Image, text_prompt: str, max_new_tokens: int = 256) -> str:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose(
            [
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        pixel_values = transform(image).unsqueeze(0).to(self._device, dtype=torch.float16 if self._device == "cuda" else torch.float32)
        question = f"<image>\n{text_prompt}"
        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
        return self._model.chat(self._tokenizer, pixel_values, question, gen_cfg).strip()

    def query_multi(self, images: list[Image.Image], text_prompt: str, max_new_tokens: int = 256) -> str:
        if len(images) == 1:
            return self.query(images[0], text_prompt, max_new_tokens=max_new_tokens)
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose(
            [
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        pixel_values = torch.stack([transform(img) for img in images]).to(
            self._device,
            dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        image_tags = "".join("<image>\n" for _ in images)
        question = f"{image_tags}{text_prompt}"
        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
        return self._model.chat(self._tokenizer, pixel_values, question, gen_cfg).strip()


class BoundaryDetector(Protocol):
    def detect_boundaries(self, video_path: str) -> list[float]: ...


@dataclass
class DetectorConfig:
    sample_fps: float = 2.0
    alpha: float = 0.4
    tpivot_grid_size: int = 5
    tpivot_iterations: int = 4
    # Reasoning VLMs (e.g. GLM-4.1V-Thinking) spend hundreds of tokens on <think> before the
    # JSON answer; 256 truncated them mid-thought so no JSON was ever emitted -> zero boundaries.
    vlm_max_new_tokens: int = 2048
    pyscenedetect_threshold: float = 27.0
    pyscenedetect_min_scene_len_frames: int = 15
    macrodata_sample_interval_sec: float = 0.5
    macrodata_tile_size_px: int = 224
    macrodata_sheet_columns: int = 5
    macrodata_sheet_rows: int = 4
    # Fine manipulation subtasks (pick/place/open/pour) are short; a 2-10s prior biased the VLM
    # toward collapsing multi-step episodes into one segment. 1-6s matches typical subtask length.
    macrodata_duration_prior_min_sec: float = 1.0
    macrodata_duration_prior_max_sec: float = 6.0
    # Macrodata must emit a full segments *list* (potentially dozens of entries) on top of any
    # <think> reasoning, unlike tpivot's single small JSON object — vlm_max_new_tokens alone is
    # too tight and truncates the answer before a single segment is ever emitted.
    #
    # 4096 was still too tight for a *reasoning* VLM: GLM-4.1V-Thinking spent the entire budget
    # inside <think> and got cut off before emitting a single segment, failing every retry. The cap
    # is only an upper bound — a model that finishes early stops at EOS — so a generous budget costs
    # a terse model (Qwen3-VL) nothing and is what a thinking model needs to reach its JSON at all.
    macrodata_vlm_max_new_tokens: int = 12288
    # A greedy VLM can deterministically return {"segments": []} on a hard video; each retry feeds
    # a stronger error-specific nudge, so a few attempts almost always break the empty response.
    macrodata_max_attempts: int = 3
    # Class E (predictive_*): see predictive_boundary.PredictiveBoundaryConfig for the semantics.
    # Kept on this shared config so the stage has a single knob surface for every detector family.
    predictive_sample_fps: float = 8.0
    predictive_clip_frames: int = 32
    predictive_clip_stride: int = 16
    predictive_horizons: tuple[int, ...] = (1, 2, 4)
    predictive_history: int = 4
    predictive_z_threshold: float = 2.0
    predictive_min_segment_s: float = 1.0
    # Class F (fusion_arc_*): ARC-Hunyuan chapter anchors + predictive infill.
    arc_model_dir: str = "/config/models/TencentARC/ARC-Hunyuan-Video-7B"
    arc_max_new_tokens: int = 1024
    # Minimum spacing enforced when admitting a predictive cut near an ARC anchor. Tuned on a
    # 50-video WGO split and validated on the held-out 50 (segF1@IoU0.5 0.5468 vs 0.4392 predictive
    # alone); the sweep 1.0-5.0 s peaked flatly at 4.0 s.
    fusion_nms_tolerance_s: float = 4.0


def predictive_config_from(cfg: DetectorConfig) -> "PredictiveBoundaryConfig":  # noqa: F821
    """Project the shared DetectorConfig onto the predictive detector's own config."""
    from cosmos_curate.pipelines.video.clipping.predictive_boundary import PredictiveBoundaryConfig

    return PredictiveBoundaryConfig(
        sample_fps=cfg.predictive_sample_fps,
        clip_frames=cfg.predictive_clip_frames,
        clip_stride=cfg.predictive_clip_stride,
        horizons=tuple(cfg.predictive_horizons),
        history=cfg.predictive_history,
        z_threshold=cfg.predictive_z_threshold,
        min_segment_s=cfg.predictive_min_segment_s,
    )


class SemanticABDBoundaryDetector:
    """Class B: semantic visual encoder + ABD-style minima detection."""

    _MODEL_MAP = {
        "clip": ("openai/clip-vit-large-patch14", "clip"),
        "siglip2": ("google/siglip2-so400m-patch14-384", "auto"),
        "dinov2": ("facebook/dinov2-large", "dinov2"),
        "dinov2_giant": ("facebook/dinov2-giant", "dinov2"),
        # DINOv3 ViT-L/16: same CLS-token extraction as DINOv2 (CLS at index 0,
        # followed by register + patch tokens), so it reuses the "dinov2" kind.
        "dinov3": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "dinov2"),
        "vjepa2": ("facebook/vjepa2-vitl-fpc64-256", "vjepa2"),
    }

    def __init__(self, encoder_key: str, cfg: DetectorConfig) -> None:
        if encoder_key not in self._MODEL_MAP:
            msg = f"Unsupported semantic encoder key: {encoder_key}"
            raise ValueError(msg)
        self.encoder_key = encoder_key
        self.cfg = cfg
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        model_id, kind = self._MODEL_MAP[self.encoder_key]
        model_path = _resolve_pretrained_path(
            model_id,
            fallbacks=("openai/clip-vit-base-patch32",) if self.encoder_key == "clip" else (),
        )
        local_only = model_path != model_id
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoProcessor,
            AutoVideoProcessor,
            CLIPModel,
            CLIPProcessor,
        )

        if kind == "clip":
            self._processor = CLIPProcessor.from_pretrained(model_path, local_files_only=local_only)
            self._model = CLIPModel.from_pretrained(model_path, local_files_only=local_only).to(self._device)
        elif kind == "dinov2":
            self._processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=local_only)
            self._model = AutoModel.from_pretrained(model_path, local_files_only=local_only).to(self._device)
        elif kind == "vjepa2":
            self._processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=local_only)
            self._model = AutoModel.from_pretrained(model_path, local_files_only=local_only).to(self._device)
        else:
            self._processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_only)
            self._model = AutoModel.from_pretrained(model_path, local_files_only=local_only).to(self._device)
        self._model.eval()

    def _encode(self, frames: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        batch_size = 16
        feats: list[npt.NDArray[np.float32]] = []
        pil_frames = [Image.fromarray(f) for f in frames]
        with torch.no_grad():
            for i in range(0, len(pil_frames), batch_size):
                b = pil_frames[i : i + batch_size]
                if self.encoder_key == "clip":
                    inputs = self._processor(images=b, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    out = self._model.get_image_features(**inputs)
                    feats.append(out.cpu().float().numpy())
                elif self.encoder_key in ("dinov2", "dinov2_giant", "dinov3"):
                    inputs = self._processor(images=b, return_tensors="pt")
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    out = self._model(**inputs).last_hidden_state[:, 0, :]
                    feats.append(out.cpu().float().numpy())
                elif self.encoder_key == "vjepa2":
                    inner: list[npt.NDArray[np.float32]] = []
                    for img in b:
                        arr = np.array(img, dtype=np.uint8)
                        clip = np.stack([arr, arr], axis=0)
                        clip_pil = [Image.fromarray(x) for x in clip]
                        inputs = self._processor(videos=[clip_pil], return_tensors="pt")
                        inputs = {k: v.to(self._device) for k, v in inputs.items()}
                        out = self._model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
                        inner.append(out.cpu().float().numpy())
                    feats.append(np.asarray(inner, dtype=np.float32))
                else:
                    inputs = self._processor(images=b, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    out = self._model.get_image_features(**inputs)
                    feats.append(out.cpu().float().numpy())
        return np.vstack(feats).astype(np.float32)

    def detect_boundaries(self, video_path: str) -> list[float]:
        frames, ts = _extract_frames_uniform(video_path, fps=self.cfg.sample_fps)
        if len(frames) < 3:
            return []
        emb = self._encode(frames)
        return _abd_boundaries(emb, ts, alpha=self.cfg.alpha)


class TPivotBoundaryDetector:
    """Class C: iterative VLM prompting (T-PIVOT style)."""

    def __init__(self, vlm: BaseVLMClient, cfg: DetectorConfig) -> None:
        self.vlm = vlm
        self.cfg = cfg

    def _prompt_transition(self, t0: float, t1: float, n_frames: int) -> str:
        return (
            f"The attached image is a contact sheet of {n_frames} video frames, arranged in a "
            "grid left-to-right then top-to-bottom in chronological order. Each tile has its "
            f"1-based frame index burned into its top-right corner. The frames span {t0:.2f}s "
            f"to {t1:.2f}s of the video.\n"
            "Identify the single tile index where the action changes (e.g. a new step, object, "
            "or motion begins). If no change is visible among these tiles, pick the tile closest "
            "to where you believe the true change lies just outside this range.\n"
            'Return only JSON: {"frame_index": <int>, "reasoning": "..."}'
        )

    def _first_pass_prompt(self, n_frames: int) -> str:
        return (
            f"The attached image is a contact sheet of {n_frames} video frames, arranged in a "
            "grid left-to-right then top-to-bottom in chronological order, covering the whole "
            "video. Each tile has its 1-based frame index burned into its top-right corner.\n"
            "List the index of every tile at which a distinct action or phase transition begins.\n"
            'Return only JSON: {"transition_frames": [..]}. Use an empty list if none.'
        )

    def detect_boundaries(self, video_path: str) -> list[float]:
        duration = _video_duration_s(video_path)  # PyAV — AV1-safe (cv2 returns 0 on AV1)
        n_frames = self.cfg.tpivot_grid_size * self.cfg.tpivot_grid_size
        frames, ts = _extract_frames_window(video_path, center_s=duration / 2.0, window_s=duration, n_frames=n_frames)
        if not frames:
            return []
        grid = _compose_grid(frames, grid_cols=self.cfg.tpivot_grid_size, frame_size=(224, 224))
        resp = self.vlm.query(grid, self._first_pass_prompt(len(frames)), max_new_tokens=self.cfg.vlm_max_new_tokens)
        idxs = _parse_json_frame_indices(resp, n_frames=len(frames))
        seeds = sorted({ts[i - 1] for i in idxs if 1 <= i <= len(ts)})
        if not seeds:
            # The whole-video grid pass is the hardest for a VLM (grid-cell -> time mapping).
            # Rather than returning zero boundaries, seed uniformly so the *local* refinement
            # passes below (an easier "where does the action change in this short window?"
            # question) still run and can place real boundaries.
            k = max(2, min(6, int(duration / 8.0)))
            seeds = [duration * (j + 0.5) / k for j in range(k)]
        refined: list[float] = []
        init_window = duration / max(len(seeds), 1)
        for s in seeds:
            center = s
            window = init_window
            for _ in range(self.cfg.tpivot_iterations):
                wf, wt = _extract_frames_window(video_path, center_s=center, window_s=window, n_frames=n_frames)
                if not wf:
                    break
                wgrid = _compose_grid(wf, grid_cols=self.cfg.tpivot_grid_size, frame_size=(224, 224))
                wr = self.vlm.query(
                    wgrid,
                    self._prompt_transition(wt[0], wt[-1], len(wf)),
                    max_new_tokens=self.cfg.vlm_max_new_tokens,
                )
                one = _parse_json_frame_indices(wr, n_frames=len(wf))
                if one:
                    idx = max(1, min(one[0], len(wt))) - 1
                    center = wt[idx]
                window = max(window / 2.0, 0.5)
            refined.append(center)
        return sorted(set(refined))


class MacrodataBoundaryDetector:
    """Class D: Macrodata subtask segmentation via timestamped contact sheets + VLM."""

    def __init__(self, vlm: BaseVLMClient, cfg: DetectorConfig) -> None:
        self.vlm = vlm
        self.cfg = cfg
        self.last_segments: list[MacrodataSegment] = []

    def _build_contact_sheets(
        self,
        video_path: str,
    ) -> tuple[list[Image.Image], float]:
        duration = _video_duration_s(video_path)
        frames, ts = _extract_frames_at_interval(video_path, self.cfg.macrodata_sample_interval_sec)
        if not frames:
            return [], duration
        tiles = [
            _burn_timestamp_tile(fr, t, self.cfg.macrodata_tile_size_px) for fr, t in zip(frames, ts, strict=True)
        ]
        per_sheet = self.cfg.macrodata_sheet_columns * self.cfg.macrodata_sheet_rows
        sheets: list[Image.Image] = []
        for i in range(0, len(tiles), per_sheet):
            chunk = tiles[i : i + per_sheet]
            sheets.append(
                _compose_macrodata_sheet(
                    chunk,
                    sheet_columns=self.cfg.macrodata_sheet_columns,
                    sheet_rows=self.cfg.macrodata_sheet_rows,
                )
            )
        return sheets, duration

    def _segment_episode(self, video_path: str) -> list[MacrodataSegment]:
        sheets, duration = self._build_contact_sheets(video_path)
        if not sheets:
            self.last_segments = []
            return []
        # Query the VLM up to macrodata_max_attempts times. Each failed attempt feeds a SPECIFIC
        # error nudge back into the prompt (the empty-segments case gets a forceful "you returned
        # zero segments, return a non-empty list" instruction — the generic "return valid JSON"
        # nudge was useless because {"segments": []} already IS valid JSON, so a greedy model just
        # repeated it). Only genuine, repeated failure raises (strict) — no whole-video fallback.
        segments: list[MacrodataSegment] | None = None
        resp = ""
        last_err: str | None = None
        for _attempt in range(max(1, self.cfg.macrodata_max_attempts)):
            prompt = _macrodata_prompt(
                duration_s=duration,
                sample_interval_sec=self.cfg.macrodata_sample_interval_sec,
                n_sheets=len(sheets),
                dur_min=self.cfg.macrodata_duration_prior_min_sec,
                dur_max=self.cfg.macrodata_duration_prior_max_sec,
                retry_error=last_err,
                attempt=_attempt,
            )
            resp = self.vlm.query_multi(sheets, prompt, max_new_tokens=self.cfg.macrodata_vlm_max_new_tokens)
            raw, parse_err = _parse_macrodata_response(resp)
            if raw is None:
                last_err = f"parse error: {parse_err}"
                continue
            segments, val_err = _validate_macrodata_segments(raw, duration)
            if val_err is not None:
                last_err = val_err
                segments = None
                continue
            break
        if not segments:
            return self._seg_fail(
                f"no valid segments after {self.cfg.macrodata_max_attempts} attempts ({last_err})",
                resp,
                video_path,
            )
        # Diagnostic: how many segments did the VLM actually return, and a snippet of its raw
        # answer. This is what tells us "under-calling model" vs "parser/validation dropped them"
        # without guessing — one concise line per video in the pipeline log.
        logger.info(
            f"[macrodata] {os.path.basename(video_path)}: dur={duration:.1f}s "
            f"-> {len(segments)} segment(s); raw={resp[:200]!r}"
        )
        self.last_segments = segments
        return segments

    def _seg_fail(self, msg: str, resp: str, video_path: str) -> list[MacrodataSegment]:
        """Handle a genuine segmentation failure (parse/validation failed after all retries).

        Default is STRICT (raise): no silent whole-video fallback — a failing video surfaces
        loudly (and names itself, so it can be pinpointed and excluded) rather than being masked
        as a fake "1 segment" result. Set env SBD_STRICT=0 to restore graceful degradation
        (return no boundaries for this video).
        """
        detail = f"{os.path.basename(video_path)}: {msg} (raw={resp[:300]!r})"
        if os.environ.get("SBD_STRICT", "1") != "0":
            raise RuntimeError(f"[macrodata] {detail}")
        logger.warning(f"[macrodata] {detail}; no boundaries for this video")
        self.last_segments = []
        return []

    def detect_boundaries(self, video_path: str) -> list[float]:
        segments = self._segment_episode(video_path)
        if len(segments) <= 1:
            return []
        return [seg.start_sec for seg in segments[1:]]


def build_boundary_detector(model_name: str, cfg: DetectorConfig) -> BoundaryDetector:
    """Factory for Class B/C/D/E/F shot boundary detectors (Class A uses native pipeline stages)."""
    if model_name.startswith("fusion_arc"):
        # Lazy, like the predictive branch: arc_fusion_boundary imports back into this module.
        from cosmos_curate.pipelines.video.clipping.arc_fusion_boundary import build_arc_fusion_detector

        return build_arc_fusion_detector(model_name, cfg)
    if model_name.startswith("predictive_"):
        # Imported lazily: predictive_boundary reuses this module's frame-decoding helpers, so a
        # module-level import here would be circular.
        from cosmos_curate.pipelines.video.clipping.predictive_boundary import build_predictive_detector

        return build_predictive_detector(model_name, predictive_config_from(cfg))
    if model_name == "semantic_clip":
        return SemanticABDBoundaryDetector("clip", cfg)
    if model_name == "semantic_siglip2":
        return SemanticABDBoundaryDetector("siglip2", cfg)
    if model_name == "semantic_dinov2":
        return SemanticABDBoundaryDetector("dinov2", cfg)
    if model_name == "semantic_dinov2_giant":
        return SemanticABDBoundaryDetector("dinov2_giant", cfg)
    if model_name == "semantic_dinov3":
        return SemanticABDBoundaryDetector("dinov3", cfg)
    if model_name == "semantic_vjepa2":
        return SemanticABDBoundaryDetector("vjepa2", cfg)
    if model_name == "tpivot_glm4v":
        return TPivotBoundaryDetector(Glm4vVLMClient(), cfg)
    if model_name == "tpivot_internvl3":
        return TPivotBoundaryDetector(InternVL3Client(), cfg)
    if model_name == "tpivot_qwen3":
        return TPivotBoundaryDetector(Qwen3VLMClient(), cfg)
    if model_name == "tpivot_molmo2":
        return TPivotBoundaryDetector(Molmo2VLMClient(), cfg)
    if model_name == "macrodata_glm4v":
        return MacrodataBoundaryDetector(Glm4vVLMClient(), cfg)
    if model_name == "macrodata_glm46v":
        # GLM-4.6V-Flash: newest GLM-V generation, same 9B class and same architecture as
        # GLM-4.1V, so it drops straight into the existing client.
        return MacrodataBoundaryDetector(Glm4vVLMClient("zai-org/GLM-4.6V-Flash"), cfg)
    if model_name == "macrodata_internvl3":
        return MacrodataBoundaryDetector(InternVL3Client(), cfg)
    if model_name == "macrodata_qwen3":
        return MacrodataBoundaryDetector(Qwen3VLMClient(), cfg)
    if model_name == "macrodata_molmo2":
        return MacrodataBoundaryDetector(Molmo2VLMClient(), cfg)
    msg = f"Unknown shot boundary detection model: {model_name}"
    raise ValueError(msg)