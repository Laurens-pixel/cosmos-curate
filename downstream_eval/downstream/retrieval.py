# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-modal retrieval metrics.

Given a query set and a gallery (e.g. captions <-> clip embeddings), measure whether
the correct counterpart is ranked highly: Recall@K, Mean Reciprocal Rank, nDCG@K.

All math is pure-numpy. ``evaluate_retrieval`` runs both text->clip and clip->text.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def _l2_normalize(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def cosine_similarity_matrix(queries: FloatArray, gallery: FloatArray) -> FloatArray:
    """Return the ``(num_queries, num_gallery)`` cosine similarity matrix."""
    q = _l2_normalize(np.asarray(queries, dtype=np.float32).reshape(len(queries), -1))
    g = _l2_normalize(np.asarray(gallery, dtype=np.float32).reshape(len(gallery), -1))
    return q @ g.T


def retrieval_metrics(
    similarity: FloatArray,
    relevant: list[list[int]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K, MRR, and nDCG@K from a similarity matrix.

    Args:
        similarity: ``(num_queries, num_gallery)`` higher-is-closer scores.
        relevant: Per-query list of relevant gallery indices (>=1 each).
        ks: Cutoffs for Recall@K and nDCG@K.

    Returns:
        Dict with ``recall@k``, ``mrr``, ``ndcg@k`` plus ``num_queries``.

    """
    similarity = np.asarray(similarity, dtype=np.float32)
    num_queries = similarity.shape[0]
    if num_queries == 0:
        return {"num_queries": 0.0}

    order = np.argsort(-similarity, axis=1)
    recall_hits = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    reciprocal_ranks = 0.0

    for i in range(num_queries):
        rel = set(relevant[i])
        ranked = order[i]
        rank_of = {int(g): int(r) for r, g in enumerate(ranked)}
        first_rank = min((rank_of[g] for g in rel if g in rank_of), default=None)
        if first_rank is not None:
            reciprocal_ranks += 1.0 / (first_rank + 1)
        for k in ks:
            top_k = set(ranked[:k].tolist())
            if rel & top_k:
                recall_hits[k] += 1
            dcg = sum(1.0 / np.log2(r + 2) for r, g in enumerate(ranked[:k]) if g in rel)
            ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), k)))
            ndcg_sums[k] += (dcg / ideal) if ideal > 0 else 0.0

    out: dict[str, float] = {"num_queries": float(num_queries), "mrr": reciprocal_ranks / num_queries}
    for k in ks:
        out[f"recall@{k}"] = recall_hits[k] / num_queries
        out[f"ndcg@{k}"] = ndcg_sums[k] / num_queries
    return out


def evaluate_retrieval(
    text_embeds: FloatArray,
    clip_embeds: FloatArray,
    relevant_text_to_clip: list[list[int]] | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    """Run text->clip and clip->text retrieval.

    When ``relevant_text_to_clip`` is omitted, a 1:1 alignment (row i of text matches
    row i of clips) is assumed.
    """
    n = len(text_embeds)
    if relevant_text_to_clip is None:
        relevant_text_to_clip = [[i] for i in range(n)]
    clip_to_text: list[list[int]] = [[] for _ in range(len(clip_embeds))]
    for text_idx, clip_idxs in enumerate(relevant_text_to_clip):
        for clip_idx in clip_idxs:
            clip_to_text[clip_idx].append(text_idx)

    sim = cosine_similarity_matrix(text_embeds, clip_embeds)
    return {
        "text_to_clip": retrieval_metrics(sim, relevant_text_to_clip, ks),
        "clip_to_text": retrieval_metrics(sim.T, clip_to_text, ks),
    }
