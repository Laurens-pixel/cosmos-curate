# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Class E architectural extensions for predictive V-JEPA 2 boundary detection.

Implements the five zero-shot variants from CLASS_E_EXTENSION.md:

* A ``native`` / ``native_joint`` — V-JEPA 2 pretrained predictor instead of linear fit
* C MC uncertainty — variance-normalized surprise via context-jitter ensemble
* D multiscale spatial error — coarse spatial grid + global cosine errors
* E adaptive stats — mean/std z-scoring with reduced ridge suppression

Shared encode + peak-picking stay in ``predictive_boundary.py``; this module owns the
anticipator / fusion differences.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from cosmos_curate.pipelines.video.clipping.predictive_boundary import (
    _EPS,
    _MIN_SCALE,
    _MIN_SIGNAL_RANGE,
    _l2_normalize,
    robust_z,
)

if TYPE_CHECKING:
    import torch
    from torch import nn

    from cosmos_curate.pipelines.video.clipping.predictive_boundary import PredictiveBoundaryConfig


def mean_std_z(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Non-robust z-score (Config E): mean/std so boundary spikes self-moderate the threshold."""
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    vals = x[finite]
    if float(np.max(np.abs(vals - float(np.mean(vals))))) < _MIN_SIGNAL_RANGE:
        return np.zeros_like(x)
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    scale = max(sd, _MIN_SCALE) if sd > _EPS else 0.0
    if scale <= 0.0:
        return np.zeros_like(x)
    out = np.zeros_like(x)
    out[finite] = (vals - mu) / scale
    return out


def z_normalize(x: npt.NDArray[np.float32], mode: str) -> npt.NDArray[np.float32]:
    if mode == "mean_std":
        return mean_std_z(x)
    return robust_z(x)


def pool_spatial_multiscale(
    tokens_t_s_d: npt.NDArray[np.float32],
    grid: int,
) -> npt.NDArray[np.float32]:
    """Pool ``(T, S, D)`` tokens into ``(T, grid*grid + 1, D)`` — regions then global mean.

    ``S`` must be a perfect square (V-JEPA 2 patch grid). ``grid=1`` returns only the global mean
    with shape ``(T, 1, D)``.
    """
    t_len, spatial, dim = tokens_t_s_d.shape
    side = int(math.sqrt(spatial))
    if side * side != spatial:
        # Fall back to global-only if layout is unexpected.
        global_pool = tokens_t_s_d.mean(axis=1, keepdims=True)
        return global_pool.astype(np.float32)
    if grid <= 1:
        return tokens_t_s_d.mean(axis=1, keepdims=True).astype(np.float32)

    patches = tokens_t_s_d.reshape(t_len, side, side, dim)
    # Uneven splits: assign leftover rows/cols to the last bin.
    row_edges = np.linspace(0, side, grid + 1, dtype=int)
    col_edges = np.linspace(0, side, grid + 1, dtype=int)
    regions: list[npt.NDArray[np.float32]] = []
    for i in range(grid):
        for j in range(grid):
            block = patches[:, row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1], :]
            regions.append(block.mean(axis=(1, 2)))
    stacked = np.stack(regions, axis=1)  # (T, G^2, D)
    global_pool = tokens_t_s_d.mean(axis=1, keepdims=True)
    return np.concatenate([stacked, global_pool], axis=1).astype(np.float32)


def _cosine_distance(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]) -> float:
    a_n = a / max(float(np.linalg.norm(a)), _EPS)
    b_n = b / max(float(np.linalg.norm(b)), _EPS)
    return float(1.0 - np.dot(a_n, b_n))


def linear_surprise_multiscale(
    region_latents: npt.NDArray[np.float32],
    *,
    horizons: tuple[int, ...],
    history: int,
    z_norm: str,
) -> npt.NDArray[np.float32]:
    """Config D + linear anticipator: per-region cosine error → max-fuse → horizon z-average.

    ``region_latents`` has shape ``(T, R, D)``.
    """
    from cosmos_curate.pipelines.video.clipping.predictive_boundary import _anticipate

    n, n_regions, _dim = region_latents.shape
    max_h = max(horizons)
    start = history + max_h - 1
    if n <= start:
        return np.full(n, np.nan, dtype=np.float32)

    per_horizon: list[npt.NDArray[np.float32]] = []
    for h in horizons:
        # Per-region raw errors, then max across regions at each t.
        fused_raw = np.full(n, np.nan, dtype=np.float32)
        for t in range(start, n):
            region_errs: list[float] = []
            for r in range(n_regions):
                hist = region_latents[t - h - history + 1 : t - h + 1, r, :]
                pred = _anticipate(hist, h)
                region_errs.append(_cosine_distance(pred, region_latents[t, r, :]))
            fused_raw[t] = float(np.max(region_errs))
        per_horizon.append(z_normalize(fused_raw, z_norm))

    fused = np.full(n, np.nan, dtype=np.float32)
    stack = np.stack(per_horizon, axis=0)[:, start:]
    fused[start:] = stack.mean(axis=0)
    return fused


def _patch_indices(tubelet: int, spatial: int, device: str) -> "torch.Tensor":
    import torch

    start = tubelet * spatial
    return torch.arange(start, start + spatial, device=device, dtype=torch.long)


def _predict_targets(
    predictor: "nn.Module",
    encoder_tokens: "torch.Tensor",
    context_tubelets: list[int],
    target_tubelets: list[int],
    spatial: int,
    *,
    drop_context_frac: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[int, npt.NDArray[np.float32]]:
    """Run V-JEPA 2 predictor once; return pooled prediction per target tubelet index.

    ``encoder_tokens``: ``(1, T*S, D)``. Context/target masks select whole tubelets of patches.
    Optional ``drop_context_frac`` randomly drops context tubelets (Config C jitter; drop_path=0
    on the released checkpoints so dropout-at-eval is a no-op).
    """
    import torch

    if not context_tubelets or not target_tubelets:
        return {}
    device = encoder_tokens.device
    ctx = list(context_tubelets)
    if drop_context_frac > 0.0 and len(ctx) > 1 and rng is not None:
        n_drop = max(1, int(round(len(ctx) * drop_context_frac)))
        n_drop = min(n_drop, len(ctx) - 1)
        drop_idx = set(rng.choice(len(ctx), size=n_drop, replace=False).tolist())
        ctx = [c for i, c in enumerate(ctx) if i not in drop_idx]
        if not ctx:
            ctx = [context_tubelets[-1]]

    ctx_idx = torch.cat([_patch_indices(t, spatial, str(device)) for t in ctx]).unsqueeze(0)
    tgt_parts = [_patch_indices(t, spatial, str(device)) for t in target_tubelets]
    tgt_idx = torch.cat(tgt_parts).unsqueeze(0)
    lengths = [spatial] * len(target_tubelets)

    with torch.no_grad():
        out = predictor(
            encoder_hidden_states=encoder_tokens,
            context_mask=[ctx_idx],
            target_mask=[tgt_idx],
        )
        pred = out.last_hidden_state[0].float().cpu().numpy()  # (sum lengths, D)

    result: dict[int, npt.NDArray[np.float32]] = {}
    offset = 0
    for tub, length in zip(target_tubelets, lengths, strict=True):
        chunk = pred[offset : offset + length]
        result[tub] = chunk.mean(axis=0).astype(np.float32)
        offset += length
    return result


def native_surprise_for_clip_tokens(  # noqa: PLR0913
    predictor: "nn.Module",
    tokens: "torch.Tensor",
    *,
    t_clip: int,
    spatial: int,
    horizons: tuple[int, ...],
    history: int,
    joint: bool,
    mc_samples: int,
    actual_pooled: npt.NDArray[np.float32],
    rng: np.random.Generator,
) -> dict[int, npt.NDArray[np.float32]]:
    """Per-horizon raw cosine errors for one clip's encoder tokens.

    Walks context endpoints ``c``; for each horizon ``h`` predicts tubelet ``c+h`` and records
    the error at that target index (same geometry as the linear ``surprise_signal``).

    * ``joint=False`` (Config A): one predictor call per horizon.
    * ``joint=True`` (Config B): one call with all valid horizon targets as mask slots.
    * ``mc_samples > 1`` (Config C): N jittered passes; error /= (pred-std + ε).
    """
    raw: dict[int, npt.NDArray[np.float32]] = {
        h: np.full(t_clip, np.nan, dtype=np.float32) for h in horizons
    }
    if t_clip < history + max(horizons):
        return raw

    n_passes = max(1, mc_samples)
    drop_frac = 0.25 if mc_samples > 1 else 0.0

    for c in range(history - 1, t_clip - 1):
        ctx = list(range(c - history + 1, c + 1))
        if ctx[0] < 0:
            continue
        valid_h = [h for h in horizons if c + h < t_clip]
        if not valid_h:
            continue

        def _preds_for(
            hs: list[int],
            *,
            drop: float,
            ctx_end: int = c,
            context: list[int] = ctx,
        ) -> dict[int, npt.NDArray[np.float32]]:
            targets = [ctx_end + h for h in hs]
            pred_by_tgt = _predict_targets(
                predictor,
                tokens,
                context,
                targets,
                spatial,
                drop_context_frac=drop,
                rng=rng,
            )
            return {h: pred_by_tgt[ctx_end + h] for h in hs if (ctx_end + h) in pred_by_tgt}

        if joint:
            passes = [_preds_for(valid_h, drop=drop_frac if n_passes > 1 else 0.0) for _ in range(n_passes)]
        else:
            passes = []
            for _ in range(n_passes):
                merged: dict[int, npt.NDArray[np.float32]] = {}
                for h in valid_h:
                    merged.update(_preds_for([h], drop=drop_frac if n_passes > 1 else 0.0))
                passes.append(merged)

        for h in valid_h:
            tgt = c + h
            pred_stack = [p[h] for p in passes if h in p]
            if not pred_stack:
                continue
            stack = np.stack(pred_stack, axis=0)
            mean_pred = stack.mean(axis=0)
            err = _cosine_distance(mean_pred, actual_pooled[tgt])
            if n_passes > 1 and stack.shape[0] > 1:
                unc = float(stack.std(axis=0).mean()) + 1e-3
                err = err / unc
            raw[h][tgt] = err

    return raw


def fuse_native_horizon_errors(
    per_horizon_raw: dict[int, npt.NDArray[np.float32]],
    per_horizon_cnt: dict[int, npt.NDArray[np.float32]],
    horizons: tuple[int, ...],
    z_norm: str,
) -> npt.NDArray[np.float32]:
    """Average overlapping contributions per horizon, z-score, then mean-fuse."""
    n = next(iter(per_horizon_cnt.values())).shape[0]
    z_list: list[npt.NDArray[np.float32]] = []
    for h in horizons:
        raw = np.full(n, np.nan, dtype=np.float32)
        cnt = per_horizon_cnt[h]
        good = cnt > 0
        raw[good] = per_horizon_raw[h][good] / cnt[good]
        z_list.append(z_normalize(raw, z_norm))
    fused = np.full(n, np.nan, dtype=np.float32)
    stack = np.stack(z_list, axis=0)
    finite = np.isfinite(stack)
    denom = finite.sum(axis=0).astype(np.float32)
    numer = np.nansum(np.where(finite, stack, 0.0), axis=0)
    good = denom > 0
    fused[good] = numer[good] / denom[good]
    return fused


def ridge_forward_gap(cfg: "PredictiveBoundaryConfig", min_gap: int) -> int:
    """Config E can shrink/remove the fixed post-boundary dead zone."""
    full = cfg.history + max(cfg.horizons) - 1
    mode = getattr(cfg, "ridge_mode", "full")
    if mode == "min_gap_only":
        return min_gap
    if mode == "scaled":
        scale = float(getattr(cfg, "ridge_scale", 0.5))
        return max(min_gap, int(round(full * scale)))
    return max(min_gap, full)
