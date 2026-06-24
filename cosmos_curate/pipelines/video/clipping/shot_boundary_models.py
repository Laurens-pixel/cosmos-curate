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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, ImageDraw
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


def _extract_frames_uniform(
    video_path: str,
    fps: float,
    *,
    max_frames: int | None = None,
) -> tuple[list[npt.NDArray[np.uint8]], list[float]]:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return [], []
    step = max(src_fps / max(fps, 1e-3), 1.0)
    idxs = [int(i * step) for i in range(int(total_frames / step))]
    if max_frames is not None:
        idxs = idxs[:max_frames]
    frames: list[npt.NDArray[np.uint8]] = []
    timestamps: list[float] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamps.append(idx / src_fps)
    cap.release()
    return frames, timestamps


def _extract_frames_window(
    video_path: str,
    center_s: float,
    window_s: float,
    n_frames: int,
) -> tuple[list[npt.NDArray[np.uint8]], list[float]]:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / src_fps) if src_fps > 0 else 0.0
    t0 = max(0.0, center_s - window_s / 2.0)
    t1 = min(duration, center_s + window_s / 2.0)
    if t1 <= t0:
        t1 = min(duration, t0 + 1.0)
    ts = np.linspace(t0, t1, n_frames).tolist()
    frames: list[npt.NDArray[np.uint8]] = []
    out_ts: list[float] = []
    for t in ts:
        idx = min(int(t * src_fps), max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out_ts.append(t)
    cap.release()
    return frames, out_ts


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
    for i, fr in enumerate(frames):
        r, c = i // grid_cols, i % grid_cols
        x = pad + c * (fw + pad)
        y = pad + r * (fh + pad)
        img = Image.fromarray(fr).resize((fw, fh), Image.Resampling.BICUBIC)
        canvas.paste(img, (x, y))
        label = str(i + 1)
        draw.rectangle([x + fw - 24, y + 2, x + fw - 2, y + 24], fill=(0, 0, 0))
        draw.text((x + fw - 20, y + 4), label, fill=(255, 255, 0))
    return canvas


def _parse_json_frame_indices(text: str, *, n_frames: int) -> list[int]:
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if "frame_index" in obj:
                idx = int(obj["frame_index"])
                return [idx] if 1 <= idx <= n_frames else []
            if "boundaries" in obj and isinstance(obj["boundaries"], list):
                out: list[int] = []
                for x in obj["boundaries"]:
                    v = int(x)
                    if 1 <= v <= n_frames:
                        out.append(v)
                return out
            if "transition_frames" in obj and isinstance(obj["transition_frames"], list):
                out = []
                for x in obj["transition_frames"]:
                    v = int(x)
                    if 1 <= v <= n_frames:
                        out.append(v)
                return out
        except Exception:
            pass
    # Fallback: parse first few integers
    out = []
    for m in re.findall(r"\b(\d+)\b", cleaned):
        v = int(m)
        if 1 <= v <= n_frames:
            out.append(v)
    return out


class BaseVLMClient(Protocol):
    def query(self, image: Image.Image, text_prompt: str, max_new_tokens: int = 256) -> str: ...


class Glm4vVLMClient:
    """GLM-4.1V-9B backend for prompt-based boundary localization."""

    MODEL_ID = "zai-org/GLM-4.1V-9B-Thinking"

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor

        model_path = _resolve_pretrained_path(self.MODEL_ID)
        local_only = model_path != self.MODEL_ID
        if model_path == self.MODEL_ID and not Path(model_path).exists():
            msg = (
                f"GLM-4.1V weights not found locally for {self.MODEL_ID}. "
                "Download to /config/models/zai-org/GLM-4.1V-9B-Thinking/ or set HF_HOME."
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


class BoundaryDetector(Protocol):
    def detect_boundaries(self, video_path: str) -> list[float]: ...


@dataclass
class DetectorConfig:
    sample_fps: float = 2.0
    alpha: float = 0.4
    tpivot_grid_size: int = 5
    tpivot_iterations: int = 4
    vlm_max_new_tokens: int = 256
    pyscenedetect_threshold: float = 27.0
    pyscenedetect_min_scene_len_frames: int = 15


class SemanticABDBoundaryDetector:
    """Class B: semantic visual encoder + ABD-style minima detection."""

    _MODEL_MAP = {
        "clip": ("openai/clip-vit-large-patch14", "clip"),
        "siglip2": ("google/siglip2-so400m-patch14-384", "auto"),
        "dinov2": ("facebook/dinov2-large", "dinov2"),
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
                elif self.encoder_key == "dinov2":
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
            f"Frames 1..{n_frames} span {t0:.2f}s to {t1:.2f}s. "
            "Identify the frame index where action changes. "
            'Return JSON: {"frame_index": <int>, "reasoning": "..."}'
        )

    def _first_pass_prompt(self, n_frames: int) -> str:
        return (
            f"Frames 1..{n_frames} cover the whole video. "
            "List every frame index where a distinct action or phase transition occurs. "
            'Return JSON: {"transition_frames": [..]}. Use an empty list if none.'
        )

    def detect_boundaries(self, video_path: str) -> list[float]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        duration = (n / fps) if fps > 0 else 0.0
        n_frames = self.cfg.tpivot_grid_size * self.cfg.tpivot_grid_size
        frames, ts = _extract_frames_window(video_path, center_s=duration / 2.0, window_s=duration, n_frames=n_frames)
        if not frames:
            return []
        grid = _compose_grid(frames, grid_cols=self.cfg.tpivot_grid_size, frame_size=(224, 224))
        resp = self.vlm.query(grid, self._first_pass_prompt(len(frames)), max_new_tokens=self.cfg.vlm_max_new_tokens)
        idxs = _parse_json_frame_indices(resp, n_frames=len(frames))
        seeds = sorted({ts[i - 1] for i in idxs if 1 <= i <= len(ts)})
        if not seeds:
            return []
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


def build_boundary_detector(model_name: str, cfg: DetectorConfig) -> BoundaryDetector:
    """Factory for Class B/C shot boundary detectors (Class A uses native pipeline stages)."""
    if model_name == "semantic_clip":
        return SemanticABDBoundaryDetector("clip", cfg)
    if model_name == "semantic_siglip2":
        return SemanticABDBoundaryDetector("siglip2", cfg)
    if model_name == "semantic_dinov2":
        return SemanticABDBoundaryDetector("dinov2", cfg)
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
    msg = f"Unknown shot boundary detection model: {model_name}"
    raise ValueError(msg)
