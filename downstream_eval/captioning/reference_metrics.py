# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-based caption metrics.

- ``cider_d``   : full CIDEr-D (tf-idf n-gram cosine with length penalty), pure-python.
- ``meteor``    : exact-match METEOR with chunk penalty (uses ``nltk`` when available for
                  full stemming/synonym matching), pure-python fallback otherwise.
- ``bertscore`` : thin wrapper over the optional ``bert_score`` package; degrades gracefully.

A "sample" is one candidate caption plus a list of reference captions.
"""

import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokenization shared by the reference metrics."""
    return _TOKEN_RE.findall(text.lower())


def _ngram_counts(tokens: list[str], max_n: int) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i : i + n])] += 1
    return counts


def cider_d(
    candidates: list[str],
    references: list[list[str]],
    max_n: int = 4,
    sigma: float = 6.0,
) -> dict[str, float]:
    """Compute corpus-level and mean per-sample CIDEr-D.

    Args:
        candidates: One predicted caption per sample.
        references: Reference caption list per sample (parallel to ``candidates``).
        max_n: Maximum n-gram order (default 4).
        sigma: Gaussian length-penalty width (default 6.0, as in the COCO implementation).

    Returns:
        Dict with ``cider_d`` (mean over samples, scaled by 10) and ``num_samples``.

    """
    if not candidates:
        return {"cider_d": 0.0, "num_samples": 0.0}

    cand_tokens = [tokenize(c) for c in candidates]
    ref_tokens = [[tokenize(r) for r in refs] for refs in references]

    document_frequency: Counter[tuple[str, ...]] = Counter()
    for refs in ref_tokens:
        present: set[tuple[str, ...]] = set()
        for ref in refs:
            present.update(_ngram_counts(ref, max_n).keys())
        for ngram in present:
            document_frequency[ngram] += 1
    log_ref_len = math.log(max(1, len(candidates)))

    def counts2vec(counts: Counter[tuple[str, ...]]) -> tuple[list[dict[tuple[str, ...], float]], list[float], int]:
        vec: list[dict[tuple[str, ...], float]] = [defaultdict(float) for _ in range(max_n)]
        norm = [0.0] * max_n
        length = 0
        for ngram, term_freq in counts.items():
            df = math.log(max(1, document_frequency[ngram]))
            n = len(ngram) - 1
            vec[n][ngram] = term_freq * (log_ref_len - df)
            norm[n] += vec[n][ngram] ** 2
            if n == 0:
                length += term_freq
        return vec, [math.sqrt(x) for x in norm], length

    def sim(
        vec_c: list[dict[tuple[str, ...], float]],
        vec_r: list[dict[tuple[str, ...], float]],
        norm_c: list[float],
        norm_r: list[float],
        len_c: int,
        len_r: int,
    ) -> list[float]:
        delta = len_c - len_r
        val = [0.0] * max_n
        for n in range(max_n):
            for ngram, count in vec_c[n].items():
                val[n] += min(count, vec_r[n].get(ngram, 0.0)) * vec_r[n].get(ngram, 0.0)
            if norm_c[n] != 0 and norm_r[n] != 0:
                val[n] /= norm_c[n] * norm_r[n]
            val[n] *= math.exp(-(delta**2) / (2 * sigma**2))
        return val

    scores: list[float] = []
    for cand, refs in zip(cand_tokens, ref_tokens, strict=True):
        vec_c, norm_c, len_c = counts2vec(_ngram_counts(cand, max_n))
        per_ref = [0.0] * max_n
        for ref in refs:
            vec_r, norm_r, len_r = counts2vec(_ngram_counts(ref, max_n))
            sims = sim(vec_c, vec_r, norm_c, norm_r, len_c, len_r)
            per_ref = [a + b for a, b in zip(per_ref, sims, strict=True)]
        score = (sum(per_ref) / max_n) / max(1, len(refs)) * 10.0
        scores.append(score)

    return {"cider_d": sum(scores) / len(scores), "num_samples": float(len(scores))}


def _meteor_single(cand: list[str], ref: list[str]) -> float:
    """Exact-match METEOR for one candidate/reference token pair."""
    if not cand or not ref:
        return 0.0
    ref_used = [False] * len(ref)
    aligned: list[tuple[int, int]] = []
    for ci, tok in enumerate(cand):
        for ri, rtok in enumerate(ref):
            if not ref_used[ri] and rtok == tok:
                ref_used[ri] = True
                aligned.append((ci, ri))
                break
    matches = len(aligned)
    if matches == 0:
        return 0.0
    precision = matches / len(cand)
    recall = matches / len(ref)
    fmean = (10 * precision * recall) / (recall + 9 * precision)
    aligned.sort()
    chunks = 1
    for k in range(1, len(aligned)):
        if aligned[k][1] != aligned[k - 1][1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return fmean * (1 - penalty)


def meteor(candidates: list[str], references: list[list[str]]) -> dict[str, float]:
    """METEOR over samples (max over references). Uses ``nltk`` when installed."""
    try:
        from nltk.translate.meteor_score import meteor_score  # type: ignore[import-untyped]

        backend = "nltk"

        def score_one(cand: str, refs: list[str]) -> float:
            return float(meteor_score([tokenize(r) for r in refs], tokenize(cand)))
    except ImportError:
        backend = "exact-match-fallback"

        def score_one(cand: str, refs: list[str]) -> float:
            cand_tok = tokenize(cand)
            return max((_meteor_single(cand_tok, tokenize(r)) for r in refs), default=0.0)

    scores = [score_one(c, r) for c, r in zip(candidates, references, strict=True)]
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"meteor": mean, "num_samples": float(len(scores)), "backend": backend}  # type: ignore[dict-item]


def bertscore(
    candidates: list[str], references: list[list[str]], lang: str = "en"
) -> dict[str, object]:
    """Compute BERTScore F1 via the optional ``bert_score`` package.

    Returns ``{"available": False, "reason": ...}`` when the dependency is missing so
    callers can skip without failing.
    """
    try:
        from bert_score import score as bert_score_fn  # type: ignore[import-untyped]
    except ImportError as exc:
        return {"available": False, "reason": f"bert_score not installed ({exc})"}

    flat_cands: list[str] = []
    flat_refs: list[list[str]] = []
    for cand, refs in zip(candidates, references, strict=True):
        flat_cands.append(cand)
        flat_refs.append(refs)
    _, _, f1 = bert_score_fn(flat_cands, flat_refs, lang=lang)
    return {"available": True, "bertscore_f1": float(f1.mean().item()), "num_samples": len(flat_cands)}
