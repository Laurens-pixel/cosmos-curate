# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable text encoders used by the retrieval and action-recognition runners.

A runner that *performs* a task (rather than only scoring precomputed results) needs to
turn text into vectors. We expose a small :class:`TextEncoder` protocol with three
concrete encoders ordered by fidelity:

1. :class:`CosmosEmbed1TextEncoder` — uses the repo's Cosmos-Embed1 text tower, so caption
   vectors live in the *same* space as the pipeline's clip video embeddings (enables true
   cross-modal text->clip retrieval / zero-shot recognition). Requires a GPU + model weights.
2. :class:`SentenceTransformerEncoder` — optional ``sentence-transformers`` backend for
   text<->text retrieval when no shared multimodal space is available.
3. :class:`HashingEncoder` — pure-numpy hashing-trick bag-of-words encoder. Always available,
   deterministic, dependency-free, so the runners (and their smoke tests) actually run
   end-to-end without any heavy stack.

``build_text_encoder("auto")`` returns the best available encoder.
"""

import re
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class TextEncoder(Protocol):
    """Encodes a list of strings into an ``(N, dim)`` float32 matrix."""

    name: str

    def encode(self, texts: list[str]) -> FloatArray:
        """Return one row vector per input string."""
        ...


class HashingEncoder:
    """Deterministic hashing-trick encoder (pure numpy, no dependencies).

    Tokens are hashed into a fixed-width vector with sign hashing and TF weighting, then
    L2-normalised. This is a genuine (if simple) text representation: lexically similar
    captions map to similar vectors, which is enough to actually run and meaningfully
    evaluate text<->text retrieval and language-based zero-shot classification.
    """

    def __init__(self, dim: int = 256, seed: int = 0) -> None:
        """Create an encoder producing ``dim``-dimensional vectors."""
        self.name = f"hashing-{dim}"
        self._dim = dim
        self._seed = seed

    def _hash(self, token: str) -> tuple[int, float]:
        h = hash((self._seed, token)) & 0xFFFFFFFF
        return h % self._dim, (1.0 if (h >> 16) & 1 else -1.0)

    def encode(self, texts: list[str]) -> FloatArray:
        """Encode ``texts`` into an L2-normalised ``(N, dim)`` matrix."""
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                idx, sign = self._hash(token)
                out[i, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


class SentenceTransformerEncoder:
    """Optional ``sentence-transformers`` backend for text<->text tasks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load a sentence-transformers model (raises if the package is missing)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = "SentenceTransformerEncoder requires 'sentence-transformers'. Install it or use HashingEncoder."
            raise RuntimeError(msg) from exc
        self.name = f"sbert-{model_name}"
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> FloatArray:
        """Encode ``texts`` with the sentence-transformers model (L2-normalised)."""
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class CosmosEmbed1TextEncoder:
    """Cosmos-Embed1 text tower, sharing the space of the pipeline's clip video embeddings.

    This is the correct encoder for *cross-modal* text->clip retrieval and zero-shot
    recognition because the pipeline's ``cosmos-embed1`` clip embeddings come from the same
    model. Requires the heavy model stack + GPU, so it is imported lazily.
    """

    def __init__(self, variant: str = "336p") -> None:
        """Instantiate and set up the Cosmos-Embed1 model (raises if unavailable)."""
        try:
            from cosmos_curate.models.cosmos_embed1 import CosmosEmbed1
        except ImportError as exc:  # pragma: no cover - needs the full pipeline env
            msg = "CosmosEmbed1TextEncoder requires the cosmos_curate model stack + GPU."
            raise RuntimeError(msg) from exc
        self.name = f"cosmos-embed1-{variant}"
        self._model = CosmosEmbed1(variant=variant, utils_only=False)
        self._model.setup()

    def encode(self, texts: list[str]) -> FloatArray:
        """Encode ``texts`` with the Cosmos-Embed1 text tower."""
        vecs = [self._model.get_text_embedding(t).reshape(-1).cpu().numpy() for t in texts]
        out = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


def build_text_encoder(kind: str = "auto", **kwargs: object) -> TextEncoder:
    """Return a text encoder by name.

    Args:
        kind: ``"hashing"``, ``"sbert"``, ``"cosmos-embed1"``, or ``"auto"`` (prefer sbert,
            then hashing — never auto-loads the GPU model).
        kwargs: Forwarded to the selected encoder constructor.

    """
    if kind == "hashing":
        return HashingEncoder(**kwargs)  # type: ignore[arg-type]
    if kind == "sbert":
        return SentenceTransformerEncoder(**kwargs)  # type: ignore[arg-type]
    if kind == "cosmos-embed1":
        return CosmosEmbed1TextEncoder(**kwargs)  # type: ignore[arg-type]
    if kind == "auto":
        try:
            return SentenceTransformerEncoder(**kwargs)  # type: ignore[arg-type]
        except RuntimeError:
            return HashingEncoder()
    msg = f"Unknown encoder kind: {kind!r}"
    raise ValueError(msg)
