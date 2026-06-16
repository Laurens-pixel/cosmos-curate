# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval *task runner*: builds the system, runs queries, then evaluates.

Unlike :mod:`downstream_eval.downstream.retrieval` (which only scores a precomputed
similarity matrix), this module actually *performs* retrieval over the curated clips:

- ``run_text_to_clip_retrieval``  : cross-modal. Query = instruction/caption text encoded by
  a shared-space encoder (Cosmos-Embed1 text tower); gallery = the pipeline's clip video
  embeddings. Answers "given text, find the right clip" in the multimodal space.
- ``run_caption_retrieval``       : text<->text. Query = GT reference captions; gallery =
  the pipeline's *generated* captions. Tests whether generated captions are discriminative
  enough to retrieve the correct clip. Runs anywhere (hashing/sbert encoder).

Both build rankings with a real nearest-neighbour search and report Recall@K / MRR / nDCG@K.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from downstream_eval.common.types import ClipRecord
from downstream_eval.downstream.encoders import TextEncoder, build_text_encoder
from downstream_eval.downstream.retrieval import cosine_similarity_matrix, retrieval_metrics

FloatArray = npt.NDArray[np.float32]


@dataclass
class RetrievalRun:
    """Result of actually running a retrieval task."""

    metrics: dict[str, float]
    similarity: FloatArray
    ranking: npt.NDArray[np.int64]
    query_ids: list[str]
    gallery_ids: list[str]

    def top_k(self, query_index: int, k: int = 5) -> list[str]:
        """Return the gallery ids ranked highest for a given query."""
        return [self.gallery_ids[j] for j in self.ranking[query_index, :k]]


def _rank(similarity: FloatArray) -> npt.NDArray[np.int64]:
    return np.argsort(-similarity, axis=1).astype(np.int64)


def run_retrieval(
    query_vectors: FloatArray,
    gallery_vectors: FloatArray,
    relevant: list[list[int]],
    query_ids: list[str],
    gallery_ids: list[str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> RetrievalRun:
    """Run nearest-neighbour retrieval and score it.

    Args:
        query_vectors: ``(Q, D)`` query embeddings.
        gallery_vectors: ``(G, D)`` gallery embeddings.
        relevant: Per-query relevant gallery indices.
        query_ids/gallery_ids: Human-readable ids for inspection.
        ks: Cutoffs for Recall@K / nDCG@K.

    """
    similarity = cosine_similarity_matrix(query_vectors, gallery_vectors)
    metrics = retrieval_metrics(similarity, relevant, ks)
    return RetrievalRun(
        metrics=metrics,
        similarity=similarity,
        ranking=_rank(similarity),
        query_ids=query_ids,
        gallery_ids=gallery_ids,
    )


def run_caption_retrieval(
    clips: list[ClipRecord],
    references: dict[str, str],
    encoder: TextEncoder | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> RetrievalRun:
    """Text<->text retrieval: GT reference captions (queries) -> generated captions (gallery).

    Args:
        clips: Curated clip records (gallery = each clip's primary generated caption).
        references: Mapping of clip-uuid -> ground-truth reference caption (the query).
        encoder: Text encoder; defaults to the best available (sbert -> hashing).
        ks: Retrieval cutoffs.

    Returns:
        A :class:`RetrievalRun`; a perfect captioner yields Recall@1 == 1.0.

    """
    encoder = encoder or build_text_encoder("auto")
    gallery_clips = [c for c in clips if c.primary_caption_or_empty()]
    gallery_ids = [c.uuid for c in gallery_clips]
    gallery_text = [c.primary_caption_or_empty() for c in gallery_clips]

    query_ids = [c.uuid for c in gallery_clips if c.uuid in references]
    query_text = [references[c.uuid] for c in gallery_clips if c.uuid in references]
    if not query_ids:
        msg = "No clip uuids in `references` matched the provided clips."
        raise ValueError(msg)

    index_of = {uid: j for j, uid in enumerate(gallery_ids)}
    relevant = [[index_of[uid]] for uid in query_ids]

    query_vecs = encoder.encode(query_text)
    gallery_vecs = encoder.encode(gallery_text)
    return run_retrieval(query_vecs, gallery_vecs, relevant, query_ids, gallery_ids, ks)


def run_text_to_clip_retrieval(
    clip_ids: list[str],
    clip_embeddings: dict[str, FloatArray],
    queries: dict[str, list[str]],
    encoder: TextEncoder,
    ks: tuple[int, ...] = (1, 5, 10),
) -> RetrievalRun:
    """Cross-modal retrieval: text queries -> clip video embeddings (shared space).

    Args:
        clip_ids: Gallery clip uuids (order defines gallery indices).
        clip_embeddings: Mapping uuid -> clip video embedding (pipeline output).
        queries: Mapping query text -> list of relevant clip uuids.
        encoder: A *shared-space* text encoder (e.g. :class:`CosmosEmbed1TextEncoder`).
            Using a non-shared encoder here is meaningless; the caller is responsible for
            pairing the right encoder with the embedding algorithm.
        ks: Retrieval cutoffs.

    """
    gallery_ids = [uid for uid in clip_ids if uid in clip_embeddings]
    gallery_vecs = np.stack([clip_embeddings[uid] for uid in gallery_ids]).astype(np.float32)
    index_of = {uid: j for j, uid in enumerate(gallery_ids)}

    query_texts = list(queries.keys())
    relevant = [[index_of[uid] for uid in queries[q] if uid in index_of] for q in query_texts]
    query_vecs = encoder.encode(query_texts)
    return run_retrieval(query_vecs, gallery_vecs, relevant, query_texts, gallery_ids, ks)
