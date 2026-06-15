# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLIPScore: reference-free vision-language alignment.

The metric *math* (``clipscore_from_embeddings``) is pure-numpy and unit-testable.
Computing CLIP embeddings from frames + text (``compute_clip_embeddings``) requires the
optional ``open_clip`` / ``torch`` stack and is loaded lazily so the rest of the suite
runs without a GPU.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def _l2_normalize(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def clipscore_from_embeddings(
    image_embeds: FloatArray, text_embeds: FloatArray, weight: float = 2.5
) -> dict[str, float]:
    """CLIPScore from paired image and text embeddings.

    ``CLIPScore = weight * max(cos(image, text), 0)`` averaged over samples
    (Hessel et al., 2021; default rescaling weight 2.5).

    Args:
        image_embeds: ``(N, D)`` image/frame embeddings.
        text_embeds: ``(N, D)`` caption embeddings (parallel to images).
        weight: Rescaling factor (default 2.5).

    Returns:
        Dict with ``clipscore`` (mean) and ``num_samples``.

    """
    image_embeds = np.asarray(image_embeds, dtype=np.float32).reshape(len(image_embeds), -1)
    text_embeds = np.asarray(text_embeds, dtype=np.float32).reshape(len(text_embeds), -1)
    if image_embeds.shape != text_embeds.shape:
        msg = f"image/text embedding shapes differ: {image_embeds.shape} vs {text_embeds.shape}"
        raise ValueError(msg)
    cos = np.sum(_l2_normalize(image_embeds) * _l2_normalize(text_embeds), axis=1)
    per_sample = weight * np.clip(cos, 0.0, None)
    return {"clipscore": float(per_sample.mean()) if len(per_sample) else 0.0, "num_samples": float(len(per_sample))}


def compute_clip_embeddings(
    frames: list[npt.NDArray[np.uint8]],
    texts: list[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> tuple[FloatArray, FloatArray]:
    """Encode frames + texts with CLIP (requires ``open_clip`` + ``torch``).

    Raises:
        RuntimeError: If the optional CLIP stack is not installed.

    """
    try:
        import open_clip  # type: ignore[import-untyped]
        import torch
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without the optional stack
        msg = "CLIPScore embedding computation requires 'open_clip' and 'torch'. Install them or pass precomputed embeddings."
        raise RuntimeError(msg) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    with torch.no_grad():
        image_batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
        image_features = model.encode_image(image_batch).cpu().numpy().astype(np.float32)
        text_features = model.encode_text(tokenizer(texts)).cpu().numpy().astype(np.float32)
    return image_features, text_features
