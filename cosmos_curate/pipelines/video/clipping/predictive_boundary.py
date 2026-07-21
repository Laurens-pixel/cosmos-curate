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
"""Predictive-surprise event boundary detection over video world-model latents.

Why this exists
---------------
The gold segments this pipeline is scored against (WGO ``segments``, AgiBot ``task_info``
action ranges, InHARD labelled actions) are **event / subtask boundaries**, not editorial
shot cuts: the footage is continuous single-camera robot and assembly video with no cuts,
fades or dissolves. The task is therefore *Generic Event Boundary Detection* (GEBD), and the
detector families already in this module answer a different question:

* ``semantic_*`` compares **adjacent frame embeddings** and thresholds the dips. One-frame
  context makes it noisy, blind to gradual transitions, and its ``mean ± alpha·std`` cutoff is
  skewed by the very outliers a boundary produces.
* ``tpivot_*`` / ``macrodata_*`` ask a VLM. Accurate-ish but ~65 (tpivot) generate() calls per
  video, and a generative model can answer ``{"segments": []}`` — a failure mode with no
  analogue in a deterministic detector.

Approach
--------
Event Segmentation Theory (Zacks et al.) holds that humans segment activity where their
**prediction of what comes next fails**. ``ESTimator`` (Jung et al., *Online Generic Event
Boundary Detection*, ICCV 2025, arXiv:2510.06855) operationalises this with a *Consistent Event
Anticipator* that predicts the next representation from prior frames, and an *Online Boundary
Discriminator* that fires where prediction error spikes relative to an adaptively-thresholded
baseline. We instantiate exactly that, with three concrete choices:

1. **Latents come from a world model.** V-JEPA 2 (Meta, 2025) is a *predictive* video model whose
   spatio-temporal tubelet latents encode dynamics, not just appearance — and whose action-
   conditioned variant was post-trained on robot manipulation, our exact domain. We encode the
   video into one latent per tubelet, so "surprise" is surprise about *motion*, which is what a
   subtask change actually is. An appearance encoder (DINOv3) is available as a complementary
   stream.
2. **The anticipator is training-free.** A local least-squares linear extrapolation over the last
   ``history`` latents models "current event dynamics" and predicts the next one. No labels, no
   fine-tuning, deterministic. (V-JEPA 2 also exposes its own learned predictor — see
   "Future work" below.)
3. **Multi-horizon surprise.** Predicting 1 step ahead catches abrupt transitions; predicting
   ``h`` steps ahead catches slow ones (a dissolve, or a gradually-changing action) that a
   1-step predictor tracks too well to notice. Per-horizon errors are converted to robust
   z-scores and averaged, so both regimes contribute on one scale.

The discriminator uses **median + MAD** rather than mean ± std: a boundary is by construction an
outlier, and outliers inflate std, raising the very threshold meant to catch them. MAD is immune.

Properties
----------
Deterministic (no sampling, no empty-list failure mode), one encoder forward per clip (no
per-boundary VLM calls), and scale-free in its single main knob (``z_threshold`` is in robust
standard deviations, so it transfers across videos and datasets without retuning).

Future work
-----------
``VJEPA2Model`` exposes ``predictor_output`` with ``context_mask`` / ``target_mask``, so the
hand-rolled linear anticipator could be replaced by the model's *own* learned predictor (context =
past tubelets, target = next tubelet). That is the most faithful reading of ESTimator's CEA, but
the mask/position-id plumbing needs GPU validation before it belongs in a production path, so it
is deliberately not shipped here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import numpy as np
import numpy.typing as npt
from loguru import logger  # type: ignore[import-not-found]

_EPS = 1e-8
# Scale factor making the median absolute deviation a consistent estimator of the standard
# deviation for normally-distributed data.
_MAD_TO_SIGMA = 1.4826
# Surprise is a cosine distance in [0, 2]; excursions below this are numerical noise, not events.
_MIN_SIGNAL_RANGE = 1e-6
# Floor on the estimated spread. When a video is predicted near-perfectly inside its events, the
# surviving variation is float round-off (~1e-7); estimating the scale *from that* and dividing by
# it manufactures enormous z-scores out of nothing. No real encoder resolves prediction differences
# below ~1e-4 cosine distance, so refuse to claim we do.
_MIN_SCALE = 1e-4
# A strict local maximum needs a sample either side of it.
_MIN_LATENTS = 3
# Fewer decoded frames than this cannot form even one tubelet pair plus context.
_MIN_FRAMES = 4


@dataclass
class PredictiveBoundaryConfig:
    """Knobs for the predictive-surprise detector.

    Defaults target manipulation footage whose gold subtasks run ~1-12 s (WGO: p10 1.3 s,
    median 3.6 s), sampled finely enough to resolve the ±0.5 s boundary tolerance the
    benchmark scores at.
    """

    # Two sampled frames form one V-JEPA 2 tubelet, so a latent spans 2/sample_fps seconds:
    # 8 fps -> 0.25 s. That resolution matters twice over — a boundary cannot be placed inside the
    # ±0.5 s scoring tolerance without it, and it shrinks the post-boundary dead zone below.
    sample_fps: float = 8.0
    # Frames per encoder forward. Must be a multiple of the tubelet size.
    clip_frames: int = 32
    # 50 % overlap: latents on a clip seam are averaged across both clips, so an encoder-boundary
    # discontinuity never masquerades as a prediction failure.
    clip_stride: int = 16
    # Prediction horizons, in latents. 1 catches abrupt transitions, larger ones catch gradual.
    horizons: tuple[int, ...] = (1, 2, 4)
    # Latents of context the anticipator fits its local linear model to.
    #
    # `history` and `max(horizons)` set the detector's temporal resolution: for
    # `history + max(horizons) - 1` latents after a boundary the anticipator's context still
    # straddles it, so the surprise decays as a ridge and no *second* boundary can be asserted
    # inside it. With the defaults that dead zone is 7 x 0.25 s = 1.75 s. Raise `sample_fps` (not
    # `history`) to resolve finer; WGO's gold subtasks are p10 1.3 s / median 3.6 s long.
    history: int = 4
    # Boundary if fused robust z-score exceeds this. In MAD-sigmas, hence dataset-portable.
    z_threshold: float = 2.0
    # Refractory period: suppress a weaker peak within this distance of a stronger one, and drop
    # boundaries hugging the video ends (a "boundary" at t=0 is not a boundary).
    min_segment_s: float = 1.0
    # Spatial pooling of tubelet patch tokens.
    spatial_pool: str = "mean"
    # Relative weights when fusing multiple encoder streams (dynamics, appearance).
    stream_weights: tuple[float, ...] = field(default=(1.0,))
    # Class E extensions (CLASS_E_EXTENSION.md) — defaults preserve the original giant baseline.
    # anticipator: "linear" | "native" (Config A) | "native_joint" (Config B)
    anticipator: str = "linear"
    # Config D: >1 keeps a G×G spatial grid (+ global) instead of a single mean-pooled latent.
    spatial_grid: int = 1
    # Config C: N>1 enables context-jitter MC uncertainty normalisation (needs native anticipator).
    mc_samples: int = 0
    # Config E: "robust" = median/MAD (baseline); "mean_std" = self-moderating non-robust z-score.
    z_norm: str = "robust"
    # Config E ridge: "full" = history+max(h)-1; "scaled" uses ridge_scale; "min_gap_only" drops it.
    ridge_mode: str = "full"
    ridge_scale: float = 0.5


# ── Pure algorithm core (numpy only; no torch, no I/O — unit-testable) ─────────────────────────


def _l2_normalize(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Row-wise L2 normalisation, safe on zero rows."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, _EPS)


def _robust_scale(x: npt.NDArray[np.float32], med: float) -> float:
    """Robust spread of ``x``, degrading gracefully as the signal becomes sparse.

    MAD is the estimator of choice — a boundary is an outlier, and outliers inflate ``std``,
    raising the very threshold meant to catch them; MAD has a 50 % breakdown point and ignores
    them. But MAD collapses to exactly zero when *most* values are identical, which is precisely
    what a very clean surprise signal looks like (near-perfect prediction inside every event,
    isolated spikes at the boundaries). Zeroing there would annihilate the only informative
    samples, so fall back through progressively less robust scales before giving up.
    """
    mad = float(np.median(np.abs(x - med))) * _MAD_TO_SIGMA
    if mad > _EPS:
        return max(mad, _MIN_SCALE)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25) / 1.349
    if iqr > _EPS:
        return max(iqr, _MIN_SCALE)
    sd = float(x.std())
    return max(sd, _MIN_SCALE) if sd > _EPS else 0.0


def robust_z(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Z-score of ``x`` against a robust centre (median) and spread (see ``_robust_scale``).

    Returns zeros for a genuinely constant signal, which correctly yields no boundaries. NaNs
    (the anticipator's warm-up window) are excluded from the statistics and scored as zero.
    """
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    vals = x[finite]
    med = float(np.median(vals))
    # The signal is a cosine distance, so its scale is absolute and comparable across videos.
    # Without this floor the `std` fallback would divide float round-off (~1e-7) by itself and
    # turn a perfectly static shot into a wall of spurious boundaries.
    if float(np.max(np.abs(vals - med))) < _MIN_SIGNAL_RANGE:
        return np.zeros_like(x)
    scale = _robust_scale(vals, med)
    if scale <= 0.0:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[finite] = (vals - med) / scale
    return out


def _anticipate(history: npt.NDArray[np.float32], horizon: int) -> npt.NDArray[np.float32]:
    """Predict the latent ``horizon`` steps past the end of ``history``.

    A per-dimension least-squares line over the history window — the training-free "Consistent
    Event Anticipator": it extrapolates the *current event's* latent-space trajectory. With a
    single history latent there is no trajectory, so it degenerates to persistence.
    """
    k = history.shape[0]
    if k == 1:
        return history[0].astype(np.float32)
    tau = np.arange(k, dtype=np.float32)
    tau_c = tau - tau.mean()
    denom = float((tau_c**2).sum())
    if denom < _EPS:  # pragma: no cover - unreachable for k >= 2
        return history[-1].astype(np.float32)
    slope = (tau_c[:, None] * (history - history.mean(axis=0, keepdims=True))).sum(axis=0) / denom
    intercept = history.mean(axis=0) - slope * tau.mean()
    return (intercept + slope * (k - 1 + horizon)).astype(np.float32)


def surprise_signal(
    latents: npt.NDArray[np.float32],
    *,
    horizons: tuple[int, ...],
    history: int,
    z_norm: str = "robust",
) -> npt.NDArray[np.float32]:
    """Fused multi-horizon predictive surprise, one value per latent (NaN where undefined).

    For each horizon ``h``, the anticipator sees the ``history`` latents ending ``h`` steps before
    ``t`` and extrapolates ``h`` steps; surprise is the cosine distance between that prediction and
    the latent actually observed at ``t``. Each horizon's raw errors are turned into robust
    z-scores independently — a 4-step-ahead prediction is inherently worse than a 1-step one, so
    the two are not comparable in raw units — and then averaged.
    """
    from cosmos_curate.pipelines.video.clipping.predictive_class_e_ext import z_normalize

    n = latents.shape[0]
    max_h = max(horizons)
    start = history + max_h - 1
    if n <= start:
        return np.full(n, np.nan, dtype=np.float32)

    z = _l2_normalize(latents.astype(np.float32))
    per_horizon: list[npt.NDArray[np.float32]] = []
    for h in horizons:
        raw = np.full(n, np.nan, dtype=np.float32)
        for t in range(start, n):
            hist = z[t - h - history + 1 : t - h + 1]
            pred = _anticipate(hist, h)
            pred /= max(float(np.linalg.norm(pred)), _EPS)
            raw[t] = 1.0 - float(np.dot(pred, z[t]))
        per_horizon.append(z_normalize(raw, z_norm))

    fused = np.full(n, np.nan, dtype=np.float32)
    stack = np.stack(per_horizon, axis=0)[:, start:]
    fused[start:] = stack.mean(axis=0)
    return fused


def fuse_streams(
    signals: list[npt.NDArray[np.float32]],
    weights: tuple[float, ...],
) -> npt.NDArray[np.float32]:
    """Weighted mean of per-stream surprise signals, ignoring NaNs elementwise."""
    if len(signals) == 1:
        return signals[0]
    w = np.asarray(weights[: len(signals)], dtype=np.float32)
    if w.sum() < _EPS:  # pragma: no cover - guarded by config validation
        w = np.ones(len(signals), dtype=np.float32)
    stack = np.stack(signals, axis=0)
    mask = np.isfinite(stack)
    wmat = np.broadcast_to(w[:, None], stack.shape) * mask
    denom = wmat.sum(axis=0)
    numer = np.nansum(np.where(mask, stack, 0.0) * wmat, axis=0)
    out = np.full(stack.shape[1], np.nan, dtype=np.float32)
    good = denom > _EPS
    out[good] = numer[good] / denom[good]
    return out


def _peak_candidates(fused: npt.NDArray[np.float32], z_threshold: float) -> list[int]:
    """Strict local maxima of ``fused`` that clear ``z_threshold`` (NaNs treated as -inf)."""
    cands: list[int] = []
    for t in range(1, fused.shape[0] - 1):
        v = fused[t]
        if not np.isfinite(v) or v < z_threshold:
            continue
        prev_v = fused[t - 1] if np.isfinite(fused[t - 1]) else -np.inf
        next_v = fused[t + 1] if np.isfinite(fused[t + 1]) else -np.inf
        if v >= prev_v and v > next_v:
            cands.append(t)
    return cands


def _suppress_ridge(
    cands: list[int],
    fused: npt.NDArray[np.float32],
    *,
    min_gap: int,
    forward_gap: int,
) -> list[int]:
    """Greedy strongest-first suppression with an asymmetric exclusion window.

    Forward of an accepted peak lies its contaminated ridge (``forward_gap``); backwards, only the
    physical ``min_gap`` applies. See ``pick_boundaries`` for why the asymmetry is the correct
    model of the anticipator's error.
    """
    accepted: list[int] = []
    for t in sorted(cands, key=lambda i: float(fused[i]), reverse=True):
        if all(abs(t - a) >= (forward_gap if t > a else min_gap) for a in accepted):
            accepted.append(t)
    return accepted


def pick_boundaries(  # noqa: PLR0913 - each knob is an independent, documented degree of freedom
    fused: npt.NDArray[np.float32],
    timestamps: npt.NDArray[np.float32],
    *,
    z_threshold: float,
    min_gap: int,
    duration_s: float,
    min_edge_s: float,
    forward_gap: int | None = None,
) -> list[float]:
    """Non-maximum-suppressed peak picking on the fused surprise signal.

    Keeps strict local maxima above ``z_threshold`` and accepts them strongest-first. Suppression
    is **asymmetric**, because the anticipator's error after a boundary is not independent of it:
    for ``history + max_horizon - 1`` latents afterwards the history window still straddles the
    boundary, so the fit stays corrupted and the signal decays as a *ridge* rather than a spike.
    Secondary maxima on that ridge are echoes of the boundary already found, not new boundaries,
    and are suppressed out to ``forward_gap``. Nothing contaminates the signal *before* a boundary,
    so backwards we only enforce the physical ``min_gap`` (from ``min_segment_s``).

    Peaks hugging either end of the video are dropped: a cut point there produces a degenerate
    sub-``min_segment_s`` clip, and true event boundaries are interior by construction.
    """
    if fused.shape[0] < _MIN_LATENTS:
        return []
    fwd = max(min_gap, forward_gap if forward_gap is not None else min_gap)
    cands = _peak_candidates(fused, z_threshold)
    if not cands:
        return []
    accepted = _suppress_ridge(cands, fused, min_gap=min_gap, forward_gap=fwd)
    return [
        round(float(timestamps[t]), 3)
        for t in sorted(accepted)
        if min_edge_s < float(timestamps[t]) < duration_s - min_edge_s
    ]


def detect_from_latents(
    streams: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
    cfg: PredictiveBoundaryConfig,
    duration_s: float,
) -> list[float]:
    """Run the full EST pipeline over one or more aligned ``(latents, timestamps)`` streams."""
    from cosmos_curate.pipelines.video.clipping.predictive_class_e_ext import (
        linear_surprise_multiscale,
        ridge_forward_gap,
    )

    if not streams:
        return []

    signals: list[npt.NDArray[np.float32]] = []
    for lat, _ in streams:
        if lat.ndim == 3:
            # Config D: (T, R, D) region latents.
            signals.append(
                linear_surprise_multiscale(
                    lat,
                    horizons=cfg.horizons,
                    history=cfg.history,
                    z_norm=cfg.z_norm,
                )
            )
        else:
            signals.append(
                surprise_signal(
                    lat,
                    horizons=cfg.horizons,
                    history=cfg.history,
                    z_norm=cfg.z_norm,
                )
            )
    if not any(np.isfinite(s).any() for s in signals):
        return []
    fused = fuse_streams(signals, cfg.stream_weights)
    timestamps = streams[0][1]

    dt = float(np.median(np.diff(timestamps))) if timestamps.shape[0] > 1 else 1.0
    min_gap = max(1, math.ceil(cfg.min_segment_s / max(dt, _EPS)))
    ridge = ridge_forward_gap(cfg, min_gap)
    return pick_boundaries(
        fused,
        timestamps,
        z_threshold=cfg.z_threshold,
        min_gap=min_gap,
        forward_gap=ridge,
        duration_s=duration_s,
        min_edge_s=cfg.min_segment_s,
    )


# ── Latent encoders ───────────────────────────────────────────────────────────────────────────


class LatentEncoder(Protocol):
    """Encodes a decoded frame sequence into temporally-ordered latents + their timestamps."""

    def encode(
        self,
        frames: list[npt.NDArray[np.uint8]],
        timestamps: list[float],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return ``(latents, latent_timestamps)`` for the given decoded frames."""
        ...


class VJepa2TubeletEncoder:
    """V-JEPA 2 spatio-temporal latents: one vector per tubelet (``tubelet_size`` frames).

    The video is encoded in overlapping clips and the per-tubelet patch tokens are pooled over
    space. Because V-JEPA 2 encodes each clip independently, clips overlap by ``clip_stride`` and
    tubelets covered twice are averaged — otherwise the latent-statistics jump at a clip seam
    reads as a prediction failure and fabricates a boundary.

    Note: the pre-existing ``semantic_vjepa2`` detector calls this same checkpoint once per frame
    on a 2-frame clip built by duplicating that frame, which discards all temporal structure (and
    costs one forward per frame). That path is left untouched; this encoder uses the model as the
    video model it is.
    """

    DEFAULT_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"

    def __init__(self, cfg: PredictiveBoundaryConfig, model_id: str | None = None) -> None:
        """Store config; the checkpoint is loaded lazily on first ``encode``.

        ``model_id`` selects the V-JEPA 2 size — ViT-L (default, ~0.3B), ViT-H (~0.7B) or ViT-g
        (~1B). All share the same architecture, tubelet size and (tubelet, patch) token layout, so
        a larger checkpoint is a true drop-in: only the encoder weights change.
        """
        self.cfg = cfg
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._tubelet = 2

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoVideoProcessor

        from cosmos_curate.pipelines.video.clipping.shot_boundary_models import _resolve_pretrained_path

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = _resolve_pretrained_path(self.model_id)
        local_only = model_path != self.model_id
        self._processor = AutoVideoProcessor.from_pretrained(model_path, local_files_only=local_only)
        model = AutoModel.from_pretrained(model_path, local_files_only=local_only).to(self._device)
        model.eval()
        self._model = model
        self._tubelet = int(getattr(model.config, "tubelet_size", 2))

    def encode(
        self,
        frames: list[npt.NDArray[np.uint8]],
        timestamps: list[float],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Encode frames into per-tubelet latents, averaging over overlapping clips.

        When ``cfg.spatial_grid > 1`` (Config D), each tubelet becomes ``(G²+1, D)`` —
        coarse spatial regions plus a global mean — so the returned latents have shape
        ``(T, R, D)`` instead of ``(T, D)``.
        """
        import torch
        from PIL import Image as PILImage

        from cosmos_curate.pipelines.video.clipping.predictive_class_e_ext import pool_spatial_multiscale

        self._load()
        assert self._model is not None
        assert self._processor is not None

        tub = self._tubelet
        n_frames = (len(frames) // tub) * tub  # drop a trailing partial tubelet
        if n_frames < tub * 2:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        frames = frames[:n_frames]
        n_tubelets = n_frames // tub
        grid = max(1, int(getattr(self.cfg, "spatial_grid", 1)))
        n_regions = 1 if grid <= 1 else grid * grid + 1

        clip_frames = max(tub * 2, (self.cfg.clip_frames // tub) * tub)
        stride = max(tub, (self.cfg.clip_stride // tub) * tub)

        acc: npt.NDArray[np.float32] | None = None
        cnt = np.zeros(n_tubelets, dtype=np.float32)

        starts = list(range(0, max(n_frames - clip_frames, 0) + 1, stride))
        if starts[-1] + clip_frames < n_frames:
            starts.append(n_frames - clip_frames)

        with torch.no_grad():
            for s in starts:
                chunk = frames[s : s + clip_frames]
                pil = [PILImage.fromarray(f) for f in chunk]
                inputs = self._processor(videos=[pil], return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                out = self._model(**inputs, skip_predictor=True).last_hidden_state  # (1, N, D)
                tokens = out[0].float().cpu().numpy()
                n_tok, dim = tokens.shape
                t_clip = len(chunk) // tub
                if t_clip <= 0 or n_tok % t_clip != 0:
                    logger.warning(f"[predictive] unexpected token count {n_tok} for {t_clip} tubelets; skipping clip")
                    continue
                spatial = n_tok // t_clip
                tok_tsd = tokens.reshape(t_clip, spatial, dim)
                pooled = pool_spatial_multiscale(tok_tsd, grid)  # (t_clip, R, D)
                if acc is None:
                    acc = np.zeros((n_tubelets, n_regions, dim), dtype=np.float32)
                base = s // tub
                acc[base : base + t_clip] += pooled
                cnt[base : base + t_clip] += 1.0

        if acc is None:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        valid = cnt > 0
        latents = acc[valid] / cnt[valid][:, None, None]
        if grid <= 1:
            latents = latents[:, 0, :]  # (T, D) — preserve baseline shape
        ts = np.asarray(
            [float(np.mean(timestamps[i * tub : (i + 1) * tub])) for i in range(n_tubelets)],
            dtype=np.float32,
        )[valid]
        return latents.astype(np.float32), ts

    def encode_native_surprise(
        self,
        frames: list[npt.NDArray[np.uint8]],
        timestamps: list[float],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Configs A/B/C: encoder forward + pretrained predictor surprise per overlapping clip."""
        import torch
        from PIL import Image as PILImage

        from cosmos_curate.pipelines.video.clipping.predictive_class_e_ext import (
            fuse_native_horizon_errors,
            native_surprise_for_clip_tokens,
        )

        self._load()
        assert self._model is not None
        assert self._processor is not None
        predictor = getattr(self._model, "predictor", None)
        if predictor is None:
            msg = (
                f"V-JEPA 2 checkpoint {self.model_id} has no .predictor submodule; "
                "Config A/B/C cannot run. Use predictive_vjepa2_giant (verified) or fall back to linear."
            )
            raise RuntimeError(msg)

        tub = self._tubelet
        n_frames = (len(frames) // tub) * tub
        if n_frames < tub * 2:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        frames = frames[:n_frames]
        n_tubelets = n_frames // tub
        clip_frames = max(tub * 2, (self.cfg.clip_frames // tub) * tub)
        stride = max(tub, (self.cfg.clip_stride // tub) * tub)
        starts = list(range(0, max(n_frames - clip_frames, 0) + 1, stride))
        if starts[-1] + clip_frames < n_frames:
            starts.append(n_frames - clip_frames)

        joint = self.cfg.anticipator == "native_joint"
        mc = int(getattr(self.cfg, "mc_samples", 0) or 0)
        rng = np.random.default_rng(0)
        per_h_sum = {h: np.zeros(n_tubelets, dtype=np.float32) for h in self.cfg.horizons}
        per_h_cnt = {h: np.zeros(n_tubelets, dtype=np.float32) for h in self.cfg.horizons}

        with torch.no_grad():
            for s in starts:
                chunk = frames[s : s + clip_frames]
                pil = [PILImage.fromarray(f) for f in chunk]
                inputs = self._processor(videos=[pil], return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                enc_out = self._model(**inputs, skip_predictor=True).last_hidden_state
                tokens = enc_out  # keep on device for predictor
                n_tok = int(tokens.shape[1])
                t_clip = len(chunk) // tub
                if t_clip <= 0 or n_tok % t_clip != 0:
                    logger.warning(f"[predictive-native] bad token layout {n_tok=} {t_clip=}")
                    continue
                spatial = n_tok // t_clip
                tok_np = tokens[0].float().cpu().numpy().reshape(t_clip, spatial, -1)
                actual_pooled = tok_np.mean(axis=1).astype(np.float32)
                clip_raw = native_surprise_for_clip_tokens(
                    predictor,
                    tokens,
                    t_clip=t_clip,
                    spatial=spatial,
                    horizons=self.cfg.horizons,
                    history=self.cfg.history,
                    joint=joint,
                    mc_samples=mc,
                    actual_pooled=actual_pooled,
                    rng=rng,
                )
                base = s // tub
                for h, arr in clip_raw.items():
                    for i, v in enumerate(arr):
                        if not np.isfinite(v):
                            continue
                        idx = base + i
                        if 0 <= idx < n_tubelets:
                            per_h_sum[h][idx] += v
                            per_h_cnt[h][idx] += 1.0

        fused = fuse_native_horizon_errors(per_h_sum, per_h_cnt, self.cfg.horizons, self.cfg.z_norm)
        ts = np.asarray(
            [float(np.mean(timestamps[i * tub : (i + 1) * tub])) for i in range(n_tubelets)],
            dtype=np.float32,
        )
        return fused, ts


class FrameCLSEncoder:
    """Per-frame appearance latents (DINOv3 / SigLIP2 CLS or image features).

    Complements the dynamics stream: a subtask change that alters *what is on screen* (a new
    object enters the gripper) shows up here even when motion continues smoothly.
    """

    _MODEL_MAP: ClassVar[dict[str, tuple[str, str]]] = {
        "dinov3": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "dinov2"),
        "dinov2": ("facebook/dinov2-large", "dinov2"),
        "siglip2": ("google/siglip2-so400m-patch14-384", "auto"),
    }

    def __init__(self, encoder_key: str, cfg: PredictiveBoundaryConfig, *, pair_frames: int = 1) -> None:
        """Store config; the checkpoint is loaded lazily on first ``encode``."""
        if encoder_key not in self._MODEL_MAP:
            msg = f"Unsupported appearance encoder: {encoder_key}"
            raise ValueError(msg)
        self.encoder_key = encoder_key
        self.cfg = cfg
        # Averaging `pair_frames` consecutive frame features puts this stream on the same temporal
        # grid as the tubelet stream, so the two can be fused elementwise.
        self.pair_frames = max(1, pair_frames)
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor

        from cosmos_curate.pipelines.video.clipping.shot_boundary_models import _resolve_pretrained_path

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id, kind = self._MODEL_MAP[self.encoder_key]
        model_path = _resolve_pretrained_path(model_id)
        local_only = model_path != model_id
        proc_cls = AutoImageProcessor if kind == "dinov2" else AutoProcessor
        self._processor = proc_cls.from_pretrained(model_path, local_files_only=local_only)
        model = AutoModel.from_pretrained(model_path, local_files_only=local_only).to(self._device)
        model.eval()
        self._model = model

    def encode(
        self,
        frames: list[npt.NDArray[np.uint8]],
        timestamps: list[float],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Encode frames into per-frame (or per-``pair_frames``-group) appearance latents."""
        import torch
        from PIL import Image as PILImage

        self._load()
        assert self._model is not None
        assert self._processor is not None

        _, kind = self._MODEL_MAP[self.encoder_key]
        pil = [PILImage.fromarray(f) for f in frames]
        feats: list[npt.NDArray[np.float32]] = []
        batch = 16
        with torch.no_grad():
            for i in range(0, len(pil), batch):
                b = pil[i : i + batch]
                inputs = self._processor(images=b, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                if kind == "dinov2":
                    out = self._model(**inputs).last_hidden_state[:, 0, :]
                else:
                    out = self._model.get_image_features(**inputs)
                feats.append(out.float().cpu().numpy())
        arr = np.vstack(feats).astype(np.float32)
        ts = np.asarray(timestamps[: arr.shape[0]], dtype=np.float32)

        p = self.pair_frames
        if p > 1:
            n = (arr.shape[0] // p) * p
            if n < p:
                return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
            arr = arr[:n].reshape(n // p, p, -1).mean(axis=1)
            ts = ts[:n].reshape(n // p, p).mean(axis=1)
        return arr.astype(np.float32), ts.astype(np.float32)


# ── Detector ──────────────────────────────────────────────────────────────────────────────────


class PredictiveSurpriseBoundaryDetector:
    """Class E: event boundaries where a world model's prediction of the next latent fails.

    Satisfies the ``BoundaryDetector`` protocol (``detect_boundaries(video_path) -> list[float]``),
    so it drops into ``ModelBoundaryClipExtractionStage`` with no downstream change. Always returns
    a concrete list — a video it cannot segment yields ``[]``, never an error.
    """

    def __init__(self, encoders: list[LatentEncoder], cfg: PredictiveBoundaryConfig) -> None:
        """Compose one or more latent streams into a single boundary detector."""
        if not encoders and cfg.anticipator != "ac_native":
            msg = "PredictiveSurpriseBoundaryDetector requires at least one encoder"
            raise ValueError(msg)
        self.encoders = encoders
        self.cfg = cfg
        self._ac_encoder = None
        self._ac_predictor = None
        self._ac_tokens_per_frame = 256
        self._ac_device = "cpu"

    def detect_boundaries(self, video_path: str) -> list[float]:
        """Return interior event-boundary timestamps, in seconds."""
        from cosmos_curate.pipelines.video.clipping.predictive_class_e_ext import ridge_forward_gap
        from cosmos_curate.pipelines.video.clipping.shot_boundary_models import (
            _extract_frames_uniform,
            _video_duration_s,
        )

        duration = _video_duration_s(video_path)
        frames, ts = _extract_frames_uniform(video_path, fps=self.cfg.sample_fps)
        if len(frames) < _MIN_FRAMES or duration <= 0:
            return []

        # Config A (AC): V-JEPA 2-AC null-action world-model surprise.
        if self.cfg.anticipator == "ac_native":
            return self._detect_ac_native(frames, ts, duration)

        # Configs A/B/C (HF): pretrained JEPA predictor path (single V-JEPA encoder stream).
        if self.cfg.anticipator in ("native", "native_joint"):
            enc = self.encoders[0]
            if not isinstance(enc, VJepa2TubeletEncoder):
                logger.error("[predictive] native anticipator requires VJepa2TubeletEncoder")
                return []
            try:
                fused, lts = enc.encode_native_surprise(frames, ts)
            except Exception as exc:  # noqa: BLE001 — never crash the stage
                logger.error(f"[predictive-native] failed on {video_path}: {exc}")
                return []
            if fused.size < _MIN_LATENTS or not np.isfinite(fused).any():
                return []
            dt = float(np.median(np.diff(lts))) if lts.shape[0] > 1 else 1.0
            min_gap = max(1, math.ceil(self.cfg.min_segment_s / max(dt, _EPS)))
            ridge = ridge_forward_gap(self.cfg, min_gap)
            return pick_boundaries(
                fused,
                lts,
                z_threshold=self.cfg.z_threshold,
                min_gap=min_gap,
                forward_gap=ridge,
                duration_s=duration,
                min_edge_s=self.cfg.min_segment_s,
            )

        streams: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []
        for enc in self.encoders:
            lat, lts = enc.encode(frames, ts)
            if lat.size and lat.shape[0] >= _MIN_LATENTS:
                streams.append((lat, lts))
        if not streams:
            return []

        # Fusing requires a shared time grid; keep only streams matching the first one's length.
        base_len = streams[0][0].shape[0]
        streams = [s for s in streams if s[0].shape[0] == base_len]

        return detect_from_latents(streams, self.cfg, duration)

    def _detect_ac_native(
        self,
        frames: list[npt.NDArray[np.uint8]],
        ts: list[float],
        duration: float,
    ) -> list[float]:
        """V-JEPA 2-AC null-action surprise → peak picking."""
        import torch

        from cosmos_curate.pipelines.video.clipping.predictive_class_e_ext import ridge_forward_gap
        from cosmos_curate.pipelines.video.clipping.vjepa2_ac import ac_surprise_signal, load_vjepa2_ac

        try:
            if self._ac_encoder is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._ac_encoder, self._ac_predictor, self._ac_tokens_per_frame = load_vjepa2_ac(device)
                self._ac_device = device
            fused, lts = ac_surprise_signal(
                self._ac_encoder,
                self._ac_predictor,
                frames,
                ts,
                horizons=self.cfg.horizons,
                history=self.cfg.history,
                tokens_per_frame=self._ac_tokens_per_frame,
                z_norm=self.cfg.z_norm,
                device=self._ac_device,
                clip_frames=min(self.cfg.clip_frames, 16),  # AC trained on short clips (~8f)
                clip_stride=max(4, self.cfg.clip_stride // 2),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[predictive-ac] failed: {exc}")
            return []
        if fused.size < _MIN_LATENTS or not np.isfinite(fused).any():
            return []
        dt = float(np.median(np.diff(lts))) if lts.shape[0] > 1 else 1.0
        min_gap = max(1, math.ceil(self.cfg.min_segment_s / max(dt, _EPS)))
        ridge = ridge_forward_gap(self.cfg, min_gap)
        return pick_boundaries(
            fused,
            lts,
            z_threshold=self.cfg.z_threshold,
            min_gap=min_gap,
            forward_gap=ridge,
            duration_s=duration,
            min_edge_s=self.cfg.min_segment_s,
        )


# V-JEPA 2 encoder sizes, weakest -> strongest. All are the same predictive (JEPA) architecture
# with identical tubelet size and token layout, so swapping size is a pure drop-in — the larger
# ones are simply stronger predictors, which is the lever for pushing IoU / boundary quality up.
_VJEPA2_SIZES = {
    "predictive_vjepa2": "facebook/vjepa2-vitl-fpc64-256",        # ViT-L, ~0.3B (default)
    "predictive_vjepa2_huge": "facebook/vjepa2-vith-fpc64-256",   # ViT-H, ~0.7B
    "predictive_vjepa2_giant": "facebook/vjepa2-vitg-fpc64-256",  # ViT-g, ~1B (strongest)
}

_GIANT = "facebook/vjepa2-vitg-fpc64-256"
_AC_GIANT = "facebook/vjepa2-ac-vitg"  # Meta hub AC weights (not HF AutoModel)

# Class E architectural variants (CLASS_E_EXTENSION.md).
_CLASS_E_VARIANTS: dict[str, dict[str, object]] = {
    # Config A: V-JEPA 2-AC pretrained world-model predictor (null-action surprise).
    # Replaces the earlier HF vitg ``.predictor`` path — AC is the robot-post-trained variant.
    "predictive_vjepa2_native_predictor": {
        "model_id": _AC_GIANT,
        "anticipator": "ac_native",
    },
    # Explicit AC alias (same settings) so job/report names can say ``ac_native``.
    "predictive_vjepa2_ac_native_predictor": {
        "model_id": _AC_GIANT,
        "anticipator": "ac_native",
    },
    # Config B: joint multi-horizon mask tokens in one HF predictor forward (vitg JEPA).
    "predictive_vjepa2_joint_horizon": {
        "model_id": _GIANT,
        "anticipator": "native_joint",
    },
    # Config C: MC context-jitter uncertainty normalisation (N=8 default).
    "predictive_vjepa2_mc_uncertainty": {
        "model_id": _GIANT,
        "anticipator": "native_joint",
        "mc_samples": 8,
    },
    # Config D: 2×2 spatial regions + global, linear anticipator.
    "predictive_vjepa2_multiscale_error": {
        "model_id": _GIANT,
        "anticipator": "linear",
        "spatial_grid": 2,
    },
    # Config E: mean/std z-scoring + halved ridge dead-zone.
    "predictive_vjepa2_adaptive_stats": {
        "model_id": _GIANT,
        "anticipator": "linear",
        "z_norm": "mean_std",
        "ridge_mode": "scaled",
        "ridge_scale": 0.5,
    },
    # Config F: adaptive stats × multiscale (best two ablations combined).
    "predictive_vjepa2_adaptive_multiscale": {
        "model_id": _GIANT,
        "anticipator": "linear",
        "spatial_grid": 2,
        "z_norm": "mean_std",
        "ridge_mode": "scaled",
        "ridge_scale": 0.5,
    },
    # Config G: adaptive stats + DINOv3 appearance stream (no VLM, still Class E).
    # Dynamics (V-JEPA2) catches motion discontinuities; DINOv3 catches object/visual changes
    # that have little motion. Adaptive stats keeps the self-moderating threshold.
    "predictive_vjepa2_adaptive_stats_fusion": {
        "model_id": _GIANT,
        "anticipator": "linear",
        "z_norm": "mean_std",
        "ridge_mode": "scaled",
        "ridge_scale": 0.5,
        "stream_weights": (1.0, 0.5),
    },
    # Config H: adaptive stats + SigLIP2 appearance stream.
    # SigLIP2 is trained with semantic alignment, so it may better catch object-identity changes
    # (e.g. new tool/object in frame) than DINOv3, which is more appearance/texture focused.
    "predictive_vjepa2_adaptive_stats_siglip2": {
        "model_id": _GIANT,
        "anticipator": "linear",
        "z_norm": "mean_std",
        "ridge_mode": "scaled",
        "ridge_scale": 0.5,
        "stream_weights": (1.0, 1.0),
    },
    # Config I: adaptive stats with no post-boundary ridge dead-zone.
    # The scaled ridge suppresses echoes of a boundary, but on WGO some gold boundaries are
    # only ~1 s apart; removing the forward dead-zone lets the detector keep valid close cuts.
    "predictive_vjepa2_adaptive_stats_min_gap": {
        "model_id": _GIANT,
        "anticipator": "linear",
        "z_norm": "mean_std",
        "ridge_mode": "min_gap_only",
    },
}


def build_predictive_detector(model_name: str, cfg: PredictiveBoundaryConfig) -> PredictiveSurpriseBoundaryDetector:
    """Build a ``predictive_*`` detector by registry name."""
    if model_name in _CLASS_E_VARIANTS:
        overrides = dict(_CLASS_E_VARIANTS[model_name])
        model_id = str(overrides.pop("model_id"))
        merged = PredictiveBoundaryConfig(**{**cfg.__dict__, **overrides})
        if merged.anticipator == "ac_native":
            # AC path loads Meta hub weights lazily in detect_boundaries — no HF encoder.
            return PredictiveSurpriseBoundaryDetector([], merged)
        # Appearance-stream fusion variants: keep V-JEPA2 dynamics as stream 0 and add a
        # per-frame appearance encoder on the same tubelet grid.
        if model_name.endswith("_adaptive_stats_fusion"):
            return PredictiveSurpriseBoundaryDetector(
                [VJepa2TubeletEncoder(merged, model_id), FrameCLSEncoder("dinov3", merged, pair_frames=2)],
                merged,
            )
        if model_name.endswith("_adaptive_stats_siglip2"):
            return PredictiveSurpriseBoundaryDetector(
                [VJepa2TubeletEncoder(merged, model_id), FrameCLSEncoder("siglip2", merged, pair_frames=2)],
                merged,
            )
        return PredictiveSurpriseBoundaryDetector([VJepa2TubeletEncoder(merged, model_id)], merged)
    if model_name in _VJEPA2_SIZES:
        return PredictiveSurpriseBoundaryDetector([VJepa2TubeletEncoder(cfg, _VJEPA2_SIZES[model_name])], cfg)
    if model_name == "predictive_dinov3":
        return PredictiveSurpriseBoundaryDetector([FrameCLSEncoder("dinov3", cfg)], cfg)
    if model_name == "predictive_fusion":
        # Both streams are placed on the tubelet grid so they fuse elementwise.
        fusion_cfg = PredictiveBoundaryConfig(**{**cfg.__dict__, "stream_weights": (1.0, 0.5)})
        return PredictiveSurpriseBoundaryDetector(
            [VJepa2TubeletEncoder(fusion_cfg), FrameCLSEncoder("dinov3", fusion_cfg, pair_frames=2)],
            fusion_cfg,
        )
    msg = f"Unknown predictive detector: {model_name}"
    raise ValueError(msg)


PREDICTIVE_MODELS = (
    *_VJEPA2_SIZES.keys(),
    *_CLASS_E_VARIANTS.keys(),
    "predictive_dinov3",
    "predictive_fusion",
)
