# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V-JEPA 2-AC (action-conditioned) world-model loader + zero-action surprise signal.

The official AC checkpoint is *not* a HuggingFace ``VJEPA2Model``; it is the Meta hub pair
``(encoder, vit_ac_predictor)`` trained on robot trajectories (DROID). For GEBD we have no
proprioception, so we condition on **zero actions / zero states** — the null-action open-loop
prior from Meta's energy-landscape notebook — and treat prediction error of the next frame's
tokens as surprise.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
from loguru import logger  # type: ignore[import-not-found]

from cosmos_curate.pipelines.video.clipping.predictive_class_e_ext import (
    _cosine_distance,
    fuse_native_horizon_errors,
)

# Default locations on this cluster (overridable via env).
_DEFAULT_AC_CKPT = Path(
    "/gpfs/work4/0/prjs0951/Sem/cosmos_curate_local_workspace/models/facebook/"
    "vjepa2-ac-vitg/vjepa2-ac-vitg.pt"
)
_DEFAULT_HUB_DIRS = (
    Path("/opt/vjepa2"),  # bind-mounted inside Apptainer jobs
    Path("/home/cmeo/.cache/torch/hub/facebookresearch_vjepa2_main"),
    Path("/gpfs/work4/0/prjs0951/Sem/torch_home/hub/facebookresearch_vjepa2_main"),
)


def _clean_backbone_key(state_dict: dict) -> dict:
    out = {}
    for key, val in state_dict.items():
        k = key.replace("module.", "").replace("backbone.", "")
        out[k] = val
    return out


def _resolve_hub_dir() -> Path:
    env = os.environ.get("VJEPA2_HUB_DIR")
    if env:
        p = Path(env)
        if (p / "src" / "hub" / "backbones.py").exists():
            return p
    for p in _DEFAULT_HUB_DIRS:
        if (p / "src" / "hub" / "backbones.py").exists():
            return p
    msg = (
        "V-JEPA 2 hub repo not found. Bind facebookresearch/vjepa2 to /opt/vjepa2 "
        "or set VJEPA2_HUB_DIR."
    )
    raise FileNotFoundError(msg)


def _resolve_ac_ckpt() -> Path:
    env = os.environ.get("VJEPA2_AC_CKPT")
    if env and Path(env).is_file():
        return Path(env)
    if _DEFAULT_AC_CKPT.is_file():
        return _DEFAULT_AC_CKPT
    msg = f"V-JEPA 2-AC checkpoint missing at {_DEFAULT_AC_CKPT} (or VJEPA2_AC_CKPT)."
    raise FileNotFoundError(msg)


def load_vjepa2_ac(device: str = "cuda") -> tuple[object, object, int]:
    """Load AC encoder + predictor; return ``(encoder, predictor, tokens_per_frame)``."""
    import torch

    hub = _resolve_hub_dir()
    ckpt_path = _resolve_ac_ckpt()
    if str(hub) not in sys.path:
        sys.path.insert(0, str(hub))

    from src.hub.backbones import _make_vjepa2_ac_model  # type: ignore[import-not-found]

    encoder, predictor = _make_vjepa2_ac_model(pretrained=False)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    encoder.load_state_dict(_clean_backbone_key(state["encoder"]), strict=False)
    predictor.load_state_dict(_clean_backbone_key(state["predictor"]), strict=True)
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    tokens_per_frame = int((256 // 16) ** 2)  # crop 256, patch 16 → 16×16
    logger.info(f"[vjepa2-ac] loaded encoder+predictor from {ckpt_path} (hub={hub})")
    return encoder, predictor, tokens_per_frame


def _frames_to_bcthw(
    frames: list[npt.NDArray[np.uint8]],
    *,
    size: int = 256,
) -> "torch.Tensor":
    """``list[(H,W,3) uint8]`` → ``[1, C, T, size, size]`` float in roughly ImageNet-norm space."""
    import torch
    import torch.nn.functional as F

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    ten = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, 3, H, W)
    b, t, c, h, w = ten.shape
    ten = ten.reshape(b * t, c, h, w)
    ten = F.interpolate(ten, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=ten.dtype, device=ten.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=ten.dtype, device=ten.device).view(1, 3, 1, 1)
    ten = (ten - mean) / std
    ten = ten.reshape(b, t, c, size, size).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
    return ten


def _encode_frames_ac(encoder, clips_bcthw, normalize_reps: bool = True):
    """Match Meta notebook ``forward_target``: each frame → 2-frame tubelet via duplication."""
    import torch
    import torch.nn.functional as F

    b, _c, t, _h, _w = clips_bcthw.size()
    # [B,C,T,H,W] → per-frame [B*T, C, 2, H, W]
    c = clips_bcthw.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(b, t, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def encode_frames_ac(encoder, clips_bcthw, normalize_reps: bool = True):
    import torch

    with torch.no_grad():
        return _encode_frames_ac(encoder, clips_bcthw, normalize_reps=normalize_reps)


def ac_surprise_signal(  # noqa: PLR0913
    encoder,
    predictor,
    frames: list[npt.NDArray[np.uint8]],
    timestamps: list[float],
    *,
    horizons: tuple[int, ...],
    history: int,
    tokens_per_frame: int,
    z_norm: str,
    device: str,
    clip_frames: int = 16,
    clip_stride: int = 8,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Zero-action multi-horizon AC surprise over overlapping windows of frames.

    Each frame is one AC timestep (duplicated into a tubelet). Surprise at target index ``c+h``
    is the cosine distance between the null-action rollout prediction and the encoded target.
    """
    import torch

    n = len(frames)
    if n < history + max(horizons) + 1:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    per_h_sum = {h: np.zeros(n, dtype=np.float32) for h in horizons}
    per_h_cnt = {h: np.zeros(n, dtype=np.float32) for h in horizons}

    starts = list(range(0, max(n - clip_frames, 0) + 1, clip_stride))
    if not starts or starts[-1] + clip_frames < n:
        starts.append(max(0, n - clip_frames))

    for s in starts:
        chunk = frames[s : s + clip_frames]
        t_clip = len(chunk)
        if t_clip < history + max(horizons):
            continue
        clips = _frames_to_bcthw(chunk).to(device)
        toks = encode_frames_ac(encoder, clips)  # [1, T*N, D]
        actual = toks[0].float().cpu().numpy().reshape(t_clip, tokens_per_frame, -1)
        actual_pooled = actual.mean(axis=1).astype(np.float32)

        for c in range(history - 1, t_clip - 1):
            # Context = history frames ending at c.
            ctx0 = c - history + 1
            if ctx0 < 0:
                continue
            z = toks[:, ctx0 * tokens_per_frame : (c + 1) * tokens_per_frame].clone()
            t_ctx = c - ctx0 + 1
            states = torch.zeros(1, t_ctx, 7, device=device, dtype=z.dtype)
            actions = torch.zeros(1, t_ctx, 7, device=device, dtype=z.dtype)

            # Autoregressive null-action rollout for each horizon.
            z_roll = z
            s_roll = states
            a_roll = actions
            for h in range(1, max(horizons) + 1):
                if c + h >= t_clip:
                    break
                pred = predictor(z_roll, a_roll, s_roll)[:, -tokens_per_frame:]
                # Layer-norm like notebook
                import torch.nn.functional as F

                pred_n = F.layer_norm(pred, (pred.size(-1),))
                pred_pool = pred_n[0].float().mean(dim=0).cpu().numpy().astype(np.float32)
                if h in per_h_sum:
                    err = _cosine_distance(pred_pool, actual_pooled[c + h])
                    idx = s + c + h
                    if 0 <= idx < n:
                        per_h_sum[h][idx] += err
                        per_h_cnt[h][idx] += 1.0
                # Append prediction as next context token block; extend action/state with zeros.
                z_roll = torch.cat([z_roll, pred_n], dim=1)
                s_roll = torch.cat(
                    [s_roll, torch.zeros(1, 1, 7, device=device, dtype=z.dtype)], dim=1
                )
                a_roll = torch.cat(
                    [a_roll, torch.zeros(1, 1, 7, device=device, dtype=z.dtype)], dim=1
                )

    fused = fuse_native_horizon_errors(per_h_sum, per_h_cnt, horizons, z_norm)
    ts = np.asarray(timestamps[:n], dtype=np.float32)
    return fused, ts
