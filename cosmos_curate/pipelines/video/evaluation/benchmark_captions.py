# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Offline caption/judge benchmark and model ranking.

Reads the JSON artefacts a pipeline run already writes — no GPU, no model weights, no
network — and aggregates them into a per-model scorecard and ranking. Point it at one or
more run directories (each ``--run NAME=/path/to/output_dir``); every run contributes one
caption model's data and they are ranked against each other.

It computes, per run:

* **Descriptive caption stats** — count, length, lexical diversity, duplicate rate.
* **Judge stats** — per-judge flag rate; precision/recall/F1/accuracy vs a manual GT
  annotation file (error-detection framing) when one is supplied.
* **Cross-judge agreement** — pairwise raw agreement + Cohen's kappa over windows judged
  by ≥2 judges (the "no cross-model agreement" gap in the old setup).
* **Multi-label GT grounding** — the temporal-grounding fix. Uses ``gt_extras.all_actions``
  (every action overlapping the window, with coverage) written by the updated AgiBot GT
  source. Reports coverage-weighted recall AND the legacy single-label recall so the
  improvement is visible, plus transition-window recall and uncovered-window rate. Falls
  back to the single GT action on older runs, so it still works on existing data.
* **Unsupervised adjacent-window consistency** — needs no GT: entity/action continuity and
  contradiction rate between consecutive window captions within a clip. This is the
  GT-free signal that sidesteps window/boundary misalignment entirely.
* **Cross-model caption similarity** — bag-of-words cosine between caption models on the
  same (video, window); shows where models agree/disagree on content.
* **Intra-clip consistency** — mean pairwise similarity of a clip's window captions.
* **Compute efficiency** — wall-clock, sec/caption, est. caption tokens/sec, sec/video,
  derived from ``processed_videos/*.json`` stage timestamps and ``summary.json``.

Usage::

    python -m cosmos_curate.pipelines.video.evaluation.benchmark_captions \\
        --run qwen=/path/agibot_qwen_output \\
        --run qwen3=/path/agibot_qwen3_output \\
        --manual-gt /path/annotation_ground_truth.json \\
        --out benchmark_report.json

Everything degrades gracefully: a metric that lacks its inputs is reported as ``null``
rather than crashing the run.
"""

import argparse
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Text helpers (dependency-free) ─────────────────────────────────────────────

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "into",
        "from",
        "with",
        "for",
        "is",
        "are",
        "be",
        "being",
        "been",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "by",
        "then",
        "while",
        "during",
        "over",
        "under",
        "near",
        "onto",
        "out",
        "up",
        "down",
        "robot",
        "robotic",
        "arm",
        "hand",
        "gripper",
        "performs",
        "performing",
        "perform",
        "shows",
        "showing",
        "video",
        "clip",
        "scene",
        "appears",
        "seems",
        "looks",
        "like",
        "towards",
        "toward",
        "while",
        "which",
        "who",
        "whose",
        "where",
        "when",
        "what",
    ]
)

_VERB_HINTS = frozenset(
    [
        "pick",
        "picks",
        "picking",
        "pick-up",
        "grasp",
        "grasps",
        "grasping",
        "grab",
        "grabs",
        "hold",
        "holds",
        "place",
        "places",
        "placing",
        "put",
        "puts",
        "putting",
        "set",
        "sets",
        "setting",
        "push",
        "pushes",
        "pushing",
        "pull",
        "pulls",
        "pulling",
        "open",
        "opens",
        "opening",
        "close",
        "closes",
        "closing",
        "move",
        "moves",
        "moving",
        "lift",
        "lifts",
        "lifting",
        "reach",
        "reaches",
        "reaching",
        "insert",
        "inserts",
        "retrieve",
        "retrieves",
        "retrieving",
        "remove",
        "removes",
        "take",
        "takes",
        "pour",
        "pours",
        "pouring",
        "drop",
        "drops",
        "cut",
        "cuts",
        "chopping",
        "chop",
        "mix",
        "mixes",
        "stir",
        "stirs",
    ]
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")


def extract_caption_text(raw: str) -> str:
    """Pull the human-readable caption out of a stored raw model output.

    Captions are saved verbatim; many models wrap the answer in a ```json fence with a
    ``{"caption": ...}`` body. Returns the plain caption string in all cases.
    """
    if not raw:
        return ""
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Try to parse a JSON object and use its "caption" field.
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for key in ("caption", "description", "text"):
                    if isinstance(obj.get(key), str) and obj[key].strip():
                        return obj[key].strip()
        except json.JSONDecodeError:
            pass
    return text


def _stem(word: str) -> str:
    """Strip common inflectional suffixes so 'picks'/'picking'/'picked' → 'pick'.

    A deliberately tiny, dependency-free stemmer — enough to align verb/noun inflections
    for overlap matching without pulling in NLTK. Not linguistically perfect, but stable.
    """
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:  # noqa: PLR2004
            return word[: -len(suf)]
    return word


def content_tokens(text: str) -> list[str]:
    """Stemmed lowercase content words (alphabetic, ≥3 chars, non-stopword)."""
    return [
        _stem(w)
        for w in (t.lower() for t in _WORD_RE.findall(text))
        if len(w) >= 3 and w not in _STOPWORDS  # noqa: PLR2004
    ]


def action_tokens(text: str) -> set[str]:
    """Verb-like tokens present in the text (for action-level matching)."""
    toks = {t.lower() for t in _WORD_RE.findall(text)}
    return toks & _VERB_HINTS


def bow_cosine(a: str, b: str) -> float:
    """Cosine similarity of two texts under a content-word bag-of-words model."""
    ca, cb = Counter(content_tokens(a)), Counter(content_tokens(b))
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in common)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap of two token sets (0 when both empty)."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def mention_recall(caption: str, action_text: str) -> float:
    """Fraction of an action phrase's content words that appear in the caption.

    Lightweight stand-in for entity/semantic matching: rewards a caption that names the
    objects/verbs of a GT action without needing an embedding model.

    NOTE: this is length-biased — a longer caption contains more words, so it matches more
    by accident. The semantic clause-level metrics (precision term) are the length-robust
    counterpart; ``length_robustness`` in the report quantifies the bias.
    """
    gt = set(content_tokens(action_text))
    if not gt:
        return 0.0
    cap = set(content_tokens(caption))
    return len(gt & cap) / len(gt)


# ── Semantic matching (optional, sentence-transformers) ────────────────────────

_CLAUSE_SPLIT_RE = re.compile(r"[.;\n]|,\s+(?:and|then)\s+|\s+and then\s+|\s+then\s+", re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """Split a caption into clauses so each *sub-action* can be matched independently.

    Splitting on sentence/clause boundaries is what makes the semantic metric length-robust:
    a clause that matches no GT action lowers precision instead of being hidden inside one
    long blob. Clauses with fewer than two content words are dropped as non-substantive.
    """
    # Keep any clause with at least one content word: a verb-only clause like "picks it up"
    # carries one content token after stop-wording yet is a genuine sub-action — dropping it
    # would collapse order detection.
    parts = _CLAUSE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if content_tokens(p)]


def build_semantic_encoder(model_path: Path) -> Any | None:  # noqa: ANN401 — returns a callable
    """Load a sentence-transformers encoder; return ``encode(list[str]) -> np.ndarray`` or None.

    Lazy/optional: if sentence-transformers (or the weights) are unavailable, returns None and
    the benchmark simply skips the semantic metrics. ``model_path`` may be a model dir or a
    HuggingFace-cache dir (``snapshots/<hash>/``) — the snapshot is resolved automatically.
    """
    try:
        import numpy as np  # noqa: PLC0415 — optional dep, imported only when semantic scoring is requested
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        print(f"[semantic] sentence-transformers unavailable ({exc}); skipping semantic metrics")
        return None

    resolved = model_path
    snaps = model_path / "snapshots"
    if snaps.is_dir():
        candidates = sorted(snaps.iterdir())
        if candidates:
            resolved = candidates[0]
    if not resolved.exists():
        print(f"[semantic] model path {resolved} not found; skipping semantic metrics")
        return None

    model = SentenceTransformer(str(resolved), device="cpu")

    def encode(texts: list[str]) -> Any:  # noqa: ANN401 — np.ndarray
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    print(f"[semantic] loaded encoder from {resolved}")
    return encode


# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass
class WindowRec:
    """One captioned window joined with its judgments + GT."""

    source_video: str
    clip_uuid: str
    window_key: str
    start_frame: int
    caption: str
    end_frame: int = 0
    judges: dict[str, dict[str, Any]] = field(default_factory=dict)  # variant -> record
    gt_action_text: str = ""
    gt_extras: dict[str, Any] = field(default_factory=dict)


def _parse_frame_range(window_key: str) -> tuple[int, int]:
    m = re.match(r"^(\d+)_(\d+)$", window_key)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _lookup_judges(judgments: dict[str, Any], source_video: str, clip_uuid: str, window_key: str) -> dict[str, Any]:
    """Return the ``{variant: record}`` dict for a window, tolerating both JSON layouts.

    Observed layouts:
      * ``source_video -> clip_uuid -> window_key -> {variant: rec}``  (current pipeline)
      * ``clip_uuid -> window_key -> {variant: rec}``                  (older merged files)
    """
    for leaf in (
        judgments.get(source_video, {}).get(clip_uuid, {}).get(window_key),
        judgments.get(clip_uuid, {}).get(window_key),
    ):
        if isinstance(leaf, dict) and leaf:
            return {k: v for k, v in leaf.items() if isinstance(v, dict)}
    return {}


def load_run(run_dir: Path) -> list[WindowRec]:
    """Join ``all_window_captions.json`` + ``all_window_judgments.json`` into WindowRecs."""
    cap_path = run_dir / "v0" / "all_window_captions.json"
    jud_path = run_dir / "v0" / "all_window_judgments.json"
    if not cap_path.exists():
        msg = f"No captions file at {cap_path}"
        raise FileNotFoundError(msg)
    captions = json.loads(cap_path.read_text())
    judgments: dict[str, Any] = json.loads(jud_path.read_text()) if jud_path.exists() else {}

    recs: list[WindowRec] = []
    for source_video, clips in captions.items():
        if not isinstance(clips, dict):
            continue
        for clip_uuid, windows in clips.items():
            if not isinstance(windows, dict):
                continue
            for window_key, raw_caption in windows.items():
                caption = extract_caption_text(raw_caption if isinstance(raw_caption, str) else "")
                judges = _lookup_judges(judgments, source_video, clip_uuid, window_key)
                gt_text, gt_extras = "", {}
                for rec in judges.values():
                    if rec.get("gt_action_text"):
                        gt_text = rec["gt_action_text"]
                        gt_extras = rec.get("gt_extras", {}) or {}
                        break
                start_f, end_f = _parse_frame_range(window_key)
                recs.append(
                    WindowRec(
                        source_video=source_video,
                        clip_uuid=clip_uuid,
                        window_key=window_key,
                        start_frame=start_f,
                        end_frame=end_f,
                        caption=caption,
                        judges=judges,
                        gt_action_text=gt_text,
                        gt_extras=gt_extras,
                    )
                )
    return recs


# ── Metric: descriptive caption stats ──────────────────────────────────────────


def caption_stats(recs: list[WindowRec]) -> dict[str, Any]:
    """Count, length distribution, lexical diversity, duplicate rate."""
    caps = [r.caption for r in recs if r.caption]
    if not caps:
        return {"n_windows": len(recs), "n_captions": 0}
    lengths = [len(c.split()) for c in caps]
    vocab: set[str] = set()
    for c in caps:
        vocab.update(content_tokens(c))
    uniq = len(set(caps))
    dup_counts = Counter(caps)
    top_dup, top_n = dup_counts.most_common(1)[0]
    return {
        "n_windows": len(recs),
        "n_captions": len(caps),
        "unique_caption_ratio": round(uniq / len(caps), 4),
        "mean_words": round(statistics.mean(lengths), 2),
        "median_words": int(statistics.median(lengths)),
        "p95_words": int(sorted(lengths)[min(len(lengths) - 1, int(0.95 * len(lengths)))]),
        "vocab_size": len(vocab),
        "most_duplicated_count": top_n,
        "most_duplicated_caption": (top_dup[:120] + "…") if len(top_dup) > 120 else top_dup,  # noqa: PLR2004
    }


# ── Metric: judges + agreement + GT precision/recall ───────────────────────────


def _verdict_is_incorrect(rec: dict[str, Any]) -> bool | None:
    """Normalise a judge record to a boolean 'flagged incorrect' (None if unparseable)."""
    v = rec.get("verdict")
    if v in ("correct",):
        return False
    if v in ("incorrect", "incorrect_object", "incorrect_action"):
        return True
    score = rec.get("score")
    if isinstance(score, (int, float)):
        return score < 5  # noqa: PLR2004 — convention: 5=correct
    return None


def judge_stats(recs: list[WindowRec], manual_gt: dict[str, bool] | None) -> dict[str, Any]:
    """Per-judge flag rate, optional P/R/F1 vs manual GT, and cross-judge agreement."""
    variants: set[str] = set()
    for r in recs:
        variants.update(r.judges)
    out: dict[str, Any] = {"judges": {}, "agreement": {}}

    for variant in sorted(variants):
        flagged = total = 0
        tp = fp = fn = tn = 0
        for r in recs:
            rec = r.judges.get(variant)
            if not rec:
                continue
            inc = _verdict_is_incorrect(rec)
            if inc is None:
                continue
            total += 1
            flagged += int(inc)
            if manual_gt is not None:
                key = f"{r.clip_uuid}|{r.window_key}"
                alt = f"{Path(r.source_video).name}|{r.window_key}"
                gt = manual_gt.get(key, manual_gt.get(alt))
                if gt is None:
                    continue
                # positive class = "incorrect" (error detection)
                if inc and gt:
                    tp += 1
                elif inc and not gt:
                    fp += 1
                elif not inc and gt:
                    fn += 1
                else:
                    tn += 1
        entry: dict[str, Any] = {
            "n_judged": total,
            "flag_rate": round(flagged / total, 4) if total else None,
        }
        if manual_gt is not None and (tp + fp + fn + tn) > 0:
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec_ = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
            acc = (tp + tn) / (tp + fp + fn + tn)
            entry.update(
                n_gt_matched=tp + fp + fn + tn,
                precision=round(prec, 4),
                recall=round(rec_, 4),
                f1=round(f1, 4),
                accuracy=round(acc, 4),
            )
        out["judges"][variant] = entry

    # Pairwise agreement + Cohen's kappa over commonly-judged windows.
    variant_list = sorted(variants)
    for i in range(len(variant_list)):
        for j in range(i + 1, len(variant_list)):
            va, vb = variant_list[i], variant_list[j]
            pairs: list[tuple[bool, bool]] = []
            for r in recs:
                ra, rb = r.judges.get(va), r.judges.get(vb)
                if not ra or not rb:
                    continue
                ia, ib = _verdict_is_incorrect(ra), _verdict_is_incorrect(rb)
                if ia is None or ib is None:
                    continue
                pairs.append((ia, ib))
            if not pairs:
                continue
            n = len(pairs)
            agree = sum(1 for a, b in pairs if a == b) / n
            # Cohen's kappa
            pa = sum(1 for a, _ in pairs if a) / n
            pb = sum(1 for _, b in pairs if b) / n
            pe = pa * pb + (1 - pa) * (1 - pb)
            kappa = (agree - pe) / (1 - pe) if (1 - pe) else 0.0
            both_inc = sum(1 for a, b in pairs if a and b) / n
            out["agreement"][f"{va}__vs__{vb}"] = {
                "n": n,
                "raw_agreement": round(agree, 4),
                "cohens_kappa": round(kappa, 4),
                "both_incorrect_rate": round(both_inc, 4),
            }
    return out


# ── Metric: multi-label GT grounding (temporal-grounding fix) ──────────────────


def _window_actions(rec: WindowRec) -> list[dict[str, Any]]:
    """Return the window's GT actions with coverage, falling back to the single label.

    New runs carry ``gt_extras.all_actions``; older runs only have ``gt_action_text``,
    which is treated as one action covering the whole window.
    """
    actions = rec.gt_extras.get("all_actions")
    if isinstance(actions, list) and actions:
        return actions
    if rec.gt_action_text:
        return [{"action_text": rec.gt_action_text, "window_coverage": 1.0, "skill": rec.gt_extras.get("gt_skill", "")}]
    return []


# An action is considered "mentioned" by a caption when this fraction of its content
# words appear in the caption. Used for the binary set-based (unweighted) metrics.
_ACTION_HIT_THRESHOLD = 0.5
# A window action is "significant" (counts toward transition/minority) above this coverage.
_SIGNIFICANT_COVERAGE = 0.15


def grounding_stats(recs: list[WindowRec], coverage_threshold: float) -> dict[str, Any]:
    """Multi-label grounding metrics — several views, side by side for comparison.

    The window's GT is the *set* of every action overlapping it (``all_actions``). We score
    a caption against that set three ways so they can be compared:

    Unweighted set view (every action counts equally — best for "the minor action must
    count too"):
      * ``action_recall_unweighted`` — mean per-action ``mention_recall`` (soft).
      * ``action_hit_rate``          — fraction of actions whose mention_recall ≥ 0.5 (binary recall).
      * ``grounded_precision``       — fraction of caption content words found in *some* action
        (penalises hallucinated content / actions not in the window).
      * ``action_f1``                — harmonic mean of action_hit_rate and grounded_precision.
      * ``boundary_recall``          — among transition windows, fraction where the caption hits
        ≥2 actions (did it capture the boundary?).

    Coverage-weighted view (temporal-importance weighting):
      * ``multilabel_coverage_recall`` — Σ(coverage_i · recall_i) / Σ(coverage_i).

    Legacy view (for direct before/after comparison):
      * ``single_label_recall``        — recall against only the majority action.
      * ``transition_minority_recall`` — recall of non-majority actions in transition windows
        (what the majority-label metric silently ignored).

    Bookkeeping:
      * ``transition_window_rate`` and ``uncovered_window_rate`` — windows whose GT actions
        cover < ``coverage_threshold`` are flagged (reported, not forced onto a label).
    """
    ml_num = ml_den = 0.0
    single_vals: list[float] = []
    minority_vals: list[float] = []
    unweighted_recall_vals: list[float] = []
    hit_rate_vals: list[float] = []
    precision_vals: list[float] = []
    boundary_hits = n_transition = n_uncovered = n_with_gt = 0

    for r in recs:
        actions = _window_actions(r)
        if not actions:
            continue
        n_with_gt += 1
        total_cov = r.gt_extras.get("window_coverage")
        if isinstance(total_cov, (int, float)) and total_cov < coverage_threshold:
            n_uncovered += 1
        # majority action = highest window_coverage (already first when from GT source)
        ordered = sorted(actions, key=lambda a: a.get("window_coverage", 0.0), reverse=True)
        single_vals.append(mention_recall(r.caption, ordered[0].get("action_text", "")))

        per_action_recall: list[float] = []
        union_gt_tokens: set[str] = set()
        for k, a in enumerate(ordered):
            cov = float(a.get("window_coverage", 1.0))
            rec_k = mention_recall(r.caption, a.get("action_text", ""))
            per_action_recall.append(rec_k)
            union_gt_tokens.update(content_tokens(a.get("action_text", "")))
            ml_num += cov * rec_k
            ml_den += cov
            if k >= 1 and cov >= _SIGNIFICANT_COVERAGE:  # minority action in a transition
                minority_vals.append(rec_k)

        # Unweighted set view — every action equal.
        unweighted_recall_vals.append(statistics.mean(per_action_recall))
        n_hit = sum(1 for v in per_action_recall if v >= _ACTION_HIT_THRESHOLD)
        hit_rate_vals.append(n_hit / len(per_action_recall))
        cap_tokens = set(content_tokens(r.caption))
        precision_vals.append(len(cap_tokens & union_gt_tokens) / len(cap_tokens) if cap_tokens else 0.0)

        is_transition = bool(r.gt_extras.get("is_transition")) or (
            sum(1 for a in actions if a.get("window_coverage", 0) >= _SIGNIFICANT_COVERAGE) >= 2  # noqa: PLR2004
        )
        if is_transition:
            n_transition += 1
            if n_hit >= 2:  # noqa: PLR2004 — captured ≥2 actions = captured the boundary
                boundary_hits += 1

    if n_with_gt == 0:
        return {"n_with_gt": 0}

    hit_rate = statistics.mean(hit_rate_vals) if hit_rate_vals else 0.0
    precision = statistics.mean(precision_vals) if precision_vals else 0.0
    f1 = (2 * hit_rate * precision / (hit_rate + precision)) if (hit_rate + precision) else 0.0
    return {
        "n_with_gt": n_with_gt,
        # unweighted set view (minor action counts equally)
        "action_recall_unweighted": round(statistics.mean(unweighted_recall_vals), 4),
        "action_hit_rate": round(hit_rate, 4),
        "grounded_precision": round(precision, 4),
        "action_f1": round(f1, 4),
        "boundary_recall": round(boundary_hits / n_transition, 4) if n_transition else None,
        # coverage-weighted view
        "multilabel_coverage_recall": round(ml_num / ml_den, 4) if ml_den else None,
        # legacy view (for comparison)
        "single_label_recall": round(statistics.mean(single_vals), 4) if single_vals else None,
        "transition_minority_recall": round(statistics.mean(minority_vals), 4) if minority_vals else None,
        # bookkeeping
        "transition_window_rate": round(n_transition / n_with_gt, 4),
        "uncovered_window_rate": round(n_uncovered / n_with_gt, 4),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; None if undefined (n<3 or zero variance)."""
    n = len(xs)
    if n < 3 or len(ys) != n:  # noqa: PLR2004
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx and dy else None


def semantic_stats(recs: list[WindowRec], encode: Any) -> dict[str, Any]:  # noqa: ANN401 — encode callable
    """Clause-level *semantic* grounding (length-robust) + a length-bias diagnostic.

    Each caption is split into clauses; each GT action is matched to its best clause by
    embedding cosine. Two scores, mirroring the lexical ones but semantic and length-aware:

    * ``sem_recall``    — mean over GT actions of the best matching clause's similarity:
      "is each action expressed somewhere in the caption?" (still grows a little with length).
    * ``sem_precision`` — mean over caption clauses of the best matching GT action's similarity:
      "is each clause the model wrote actually supported?" A verbose caption padded with
      clauses that match no action is **penalised** here — this is what stops long captions
      from winning.
    * ``sem_f1``        — harmonic mean; the length-robust headline.

    ``length_robustness`` reports Pearson correlation between caption word-count and each
    score across windows. A score that a longer caption can game shows high positive
    correlation; ``sem_f1`` should be markedly lower than lexical recall — printed so you can
    confirm it visually.
    """
    import numpy as np  # noqa: PLC0415 — optional dep, imported only when semantic scoring is requested

    # Per-window inputs
    rows: list[tuple[list[str], list[str], int, float]] = []  # clauses, action_texts, word_count, lexical_recall
    all_texts: set[str] = set()
    for r in recs:
        actions = _window_actions(r)
        if not actions or not r.caption:
            continue
        clauses = split_clauses(r.caption) or [r.caption]
        action_texts = [str(a.get("action_text", "")) for a in actions if a.get("action_text")]
        if not action_texts:
            continue
        lexical_recall = statistics.mean(mention_recall(r.caption, a) for a in action_texts)
        rows.append((clauses, action_texts, len(r.caption.split()), lexical_recall))
        all_texts.update(clauses)
        all_texts.update(action_texts)

    if not rows:
        return {}

    # One batched embedding pass for all unique texts.
    text_list = sorted(all_texts)
    vecs = encode(text_list)
    emb = {t: vecs[i] for i, t in enumerate(text_list)}

    sem_recalls: list[float] = []
    sem_precisions: list[float] = []
    sem_f1s: list[float] = []
    word_counts: list[int] = []
    lexical_recalls: list[float] = []
    for clauses, action_texts, wc, lex_rec in rows:
        cvecs = np.stack([emb[c] for c in clauses])
        avecs = np.stack([emb[a] for a in action_texts])
        sim = cvecs @ avecs.T  # (clauses, actions); vectors are L2-normalised
        recall = float(sim.max(axis=0).mean())  # best clause per action
        precision = float(sim.max(axis=1).mean())  # best action per clause
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
        sem_recalls.append(recall)
        sem_precisions.append(precision)
        sem_f1s.append(f1)
        word_counts.append(wc)
        lexical_recalls.append(lex_rec)

    return {
        "sem_recall": round(statistics.mean(sem_recalls), 4),
        "sem_precision": round(statistics.mean(sem_precisions), 4),
        "sem_f1": round(statistics.mean(sem_f1s), 4),
        "n_scored": len(sem_f1s),
        "length_robustness": {
            # |r| near 0 = length-robust; high positive = a longer caption inflates the score
            "lexical_recall_vs_length_r": _pearson([float(w) for w in word_counts], lexical_recalls),
            "sem_recall_vs_length_r": _pearson([float(w) for w in word_counts], sem_recalls),
            "sem_f1_vs_length_r": _pearson([float(w) for w in word_counts], sem_f1s),
            "mean_caption_words": round(statistics.mean(word_counts), 1),
        },
    }


# ── Extended grounding: granularity, order, idle, robustness ───────────────────

# A caption that asserts "nothing is happening" — credited on genuinely idle windows.
_IDLE_RE = re.compile(
    r"\b(no(?:thing| (?:action|activity|significant|movement|motion|change|object|interaction))"
    r"|idle|stationary|remains? (?:still|stationary|idle)|stays? still|waiting|does not (?:move|interact)"
    r"|not? interact|no one|empty)\b",
    re.IGNORECASE,
)


def is_idle_caption(caption: str) -> bool:
    """Return True if the caption asserts that no real action occurs."""
    return bool(_IDLE_RE.search(caption))


def bidirectional_containment(a: str, b: str) -> float:
    """Granularity-aware lexical match: |A∩B| / min(|A|,|B|) on content tokens.

    Full credit when the shorter phrase is contained in the longer one, so a *coarse*
    caption ("vegetable") matching a *specific* GT ("purple eggplant") — and the reverse —
    both score. Plain ``mention_recall`` only credits the GT→caption direction, so it
    under-credits captions that are correct but less specific (or more specific) than GT.
    """
    sa, sb = set(content_tokens(a)), set(content_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def granularity_stats(recs: list[WindowRec]) -> dict[str, Any]:
    """Recall under granularity-aware (bidirectional containment) matching vs strict.

    The gap between ``granularity_aware_recall`` and the strict ``single_label_recall``
    shows how much a model is penalised purely for naming objects at a different
    granularity than the GT (coarse↔fine), not for being wrong.
    """
    aware: list[float] = []
    for r in recs:
        actions = _window_actions(r)
        if not actions or not r.caption:
            continue
        aware.append(statistics.mean(bidirectional_containment(r.caption, a.get("action_text", "")) for a in actions))
    return {"granularity_aware_recall": round(statistics.mean(aware), 4)} if aware else {}


def idle_stats(recs: list[WindowRec], coverage_threshold: float) -> dict[str, Any]:
    """Score idle/no-action windows separately so 'nothing happening' is judged on its merits.

    A window is treated as *idle* when it has no GT action or its ``window_coverage`` is below
    ``coverage_threshold``. Reports whether captions correctly say so, and how often captions
    falsely claim idleness on active windows (a failure mode for over-cautious models).
    """
    idle_total = idle_correct = 0
    active_total = active_false_idle = 0
    for r in recs:
        if not r.caption:
            continue
        actions = _window_actions(r)
        cov = r.gt_extras.get("window_coverage")
        is_idle_window = (not actions) or (isinstance(cov, (int, float)) and cov < coverage_threshold)
        if is_idle_window:
            idle_total += 1
            idle_correct += int(is_idle_caption(r.caption))
        else:
            active_total += 1
            active_false_idle += int(is_idle_caption(r.caption))
    if idle_total == 0 and active_total == 0:
        return {}
    return {
        "n_idle_windows": idle_total,
        "idle_caption_recall": round(idle_correct / idle_total, 4) if idle_total else None,
        "false_idle_rate_on_active": round(active_false_idle / active_total, 4) if active_total else None,
    }


def _concordance(positions: list[int | None]) -> float | None:
    """Fraction of ordered pairs whose matched clause positions are non-decreasing.

    ``positions[i]`` is the caption-clause index that best matches the i-th GT action (GT
    actions already sorted in temporal order); ``None`` = that action wasn't matched. With
    fewer than two matched actions the order is undefined → None.
    """
    idx = [(i, p) for i, p in enumerate(positions) if p is not None]
    if len(idx) < 2:  # noqa: PLR2004
        return None
    pairs = concordant = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs += 1
            if idx[a][1] <= idx[b][1]:  # caption mentions them in the GT order
                concordant += 1
    return concordant / pairs if pairs else None


def order_stats(recs: list[WindowRec], encode: Any | None = None) -> dict[str, Any]:  # noqa: ANN401
    """Score temporal-order correctness: are actions mentioned in the GT order.

    For windows with ≥2 temporally-ordered GT actions, each action is aligned to its
    best-matching caption clause; ``order_consistency`` is the fraction of action pairs the
    caption presents in the correct order. A caption saying "places then picks" when GT is
    pick→place scores low here even if its bag-of-words recall is perfect. Computed lexically,
    and semantically too when an encoder is supplied.
    """
    lex_vals: list[float] = []
    sem_vals: list[float] = []
    n_multi = 0

    # Pre-embed clauses + actions once if semantic requested.
    emb: dict[str, Any] = {}
    if encode is not None:
        texts: set[str] = set()
        for r in recs:
            actions = _window_actions(r)
            if len([a for a in actions if a.get("action_text")]) >= 2 and r.caption:  # noqa: PLR2004
                texts.update(split_clauses(r.caption) or [r.caption])
                texts.update(a.get("action_text", "") for a in actions)
        text_list = sorted(t for t in texts if t)
        if text_list:
            vecs = encode(text_list)
            emb = {t: vecs[i] for i, t in enumerate(text_list)}

    for r in recs:
        actions = [a for a in _window_actions(r) if a.get("action_text")]
        if len(actions) < 2 or not r.caption:  # noqa: PLR2004
            continue
        ordered = sorted(actions, key=lambda a: a.get("start_frame", 0))
        clauses = split_clauses(r.caption) or [r.caption]
        n_multi += 1

        lex_pos: list[int | None] = []
        for a in ordered:
            scores = [bidirectional_containment(c, a.get("action_text", "")) for c in clauses]
            lex_pos.append(max(range(len(clauses)), key=scores.__getitem__) if max(scores) > 0 else None)
        c = _concordance(lex_pos)
        if c is not None:
            lex_vals.append(c)

        if emb:
            import numpy as np  # noqa: PLC0415 — only when semantic order requested

            cvecs = np.stack([emb[x] for x in clauses if x in emb]) if any(x in emb for x in clauses) else None
            sem_pos: list[int | None] = []
            valid_clauses = [x for x in clauses if x in emb]
            for a in ordered:
                av = emb.get(a.get("action_text", ""))
                if av is None or cvecs is None or len(valid_clauses) == 0:
                    sem_pos.append(None)
                    continue
                sims = cvecs @ av
                sem_pos.append(int(sims.argmax()) if float(sims.max()) > 0 else None)
            c2 = _concordance(sem_pos)
            if c2 is not None:
                sem_vals.append(c2)

    if n_multi == 0:
        return {}
    out: dict[str, Any] = {
        "n_ordered_windows": n_multi,
        "order_consistency_lexical": round(statistics.mean(lex_vals), 4) if lex_vals else None,
    }
    if encode is not None:
        out["order_consistency_semantic"] = round(statistics.mean(sem_vals), 4) if sem_vals else None
    return out


def robustness_stats(recs: list[WindowRec], thresholds: tuple[float, ...] = (0.3, 0.5, 0.7)) -> dict[str, Any]:
    """Threshold sensitivity of the binary hit-rate + per-window action-count distribution.

    ``hit_rate@τ`` is recomputed at several match thresholds so a reader sees how much the
    headline depends on the (otherwise arbitrary) 0.5 cutoff. ``actions_per_window`` exposes
    how often windows span 1 / 2 / 3+ GT actions — i.e. how much the current segmentation
    forces multi-action windows onto the caption metric.
    """
    by_threshold: dict[str, float | None] = {}
    for t in thresholds:
        vals: list[float] = []
        for r in recs:
            actions = _window_actions(r)
            if not actions or not r.caption:
                continue
            per = [mention_recall(r.caption, a.get("action_text", "")) for a in actions]
            vals.append(sum(1 for v in per if v >= t) / len(per))
        by_threshold[f"hit_rate@{t}"] = round(statistics.mean(vals), 4) if vals else None

    buckets = Counter()
    for r in recs:
        n = len(_window_actions(r))
        if n:
            buckets[min(n, 3)] += 1
    total = sum(buckets.values())
    dist = (
        {
            "windows_1_action": buckets.get(1, 0),
            "windows_2_actions": buckets.get(2, 0),
            "windows_3plus_actions": buckets.get(3, 0),
            "mean_actions_per_window": round(
                sum(len(_window_actions(r)) for r in recs if _window_actions(r)) / total, 3
            ),
        }
        if total
        else {}
    )
    return {"threshold_sensitivity": by_threshold, "actions_per_window": dist}


# ── Metric: unsupervised adjacent-window consistency ───────────────────────────


def consistency_stats(recs: list[WindowRec]) -> dict[str, Any]:
    """Entity/action continuity + contradiction rate between consecutive windows.

    Needs no GT. For each clip, sort windows by start frame and compare each caption with
    the next: object continuity (content-word Jaccard), action overlap (verb Jaccard), and
    a contradiction flag (clip's captions disagree on the primary object with zero overlap).
    Also reports intra-clip consistency = mean pairwise BoW cosine across a clip's windows.
    """
    by_clip: dict[str, list[WindowRec]] = defaultdict(list)
    for r in recs:
        by_clip[r.clip_uuid].append(r)

    obj_cont: list[float] = []
    act_cont: list[float] = []
    contradictions = adjacent_pairs = 0
    intra_clip_means: list[float] = []

    for clip_recs in by_clip.values():
        clip_recs.sort(key=lambda r: r.start_frame)
        caps = [r.caption for r in clip_recs if r.caption]
        # intra-clip consistency: mean pairwise cosine
        if len(caps) >= 2:  # noqa: PLR2004
            sims = [bow_cosine(caps[i], caps[k]) for i in range(len(caps)) for k in range(i + 1, len(caps))]
            if sims:
                intra_clip_means.append(statistics.mean(sims))
        # adjacent continuity
        for a, b in itertools.pairwise(clip_recs):
            if not a.caption or not b.caption:
                continue
            adjacent_pairs += 1
            oa, ob = set(content_tokens(a.caption)), set(content_tokens(b.caption))
            oj = jaccard(oa, ob)
            obj_cont.append(oj)
            act_cont.append(jaccard(action_tokens(a.caption), action_tokens(b.caption)))
            # contradiction heuristic: zero content overlap between neighbours that are
            # both non-trivial → the narration jumps with no shared entity/action.
            if oj == 0.0 and len(oa) >= 2 and len(ob) >= 2:  # noqa: PLR2004
                contradictions += 1

    if adjacent_pairs == 0:
        return {"adjacent_pairs": 0, "intra_clip_consistency": None}
    return {
        "adjacent_pairs": adjacent_pairs,
        "object_continuity": round(statistics.mean(obj_cont), 4),
        "action_continuity": round(statistics.mean(act_cont), 4),
        "contradiction_rate": round(contradictions / adjacent_pairs, 4),
        "intra_clip_consistency": round(statistics.mean(intra_clip_means), 4) if intra_clip_means else None,
    }


# ── Metric: compute efficiency ─────────────────────────────────────────────────


def efficiency_stats(run_dir: Path, recs: list[WindowRec]) -> dict[str, Any]:
    """Wall-clock, throughput and est. caption tokens/sec from run artefacts."""
    pv_dir = run_dir / "processed_videos"
    summary_path = run_dir / "summary.json"
    out: dict[str, Any] = {}

    starts: list[float] = []
    ends: list[float] = []
    # Per-stage compute summed across videos. ``stage_timestamps`` is flat:
    # {"VllmCaptionStage_start": ts, "VllmCaptionStage_end": ts, ...}. This is true
    # compute time per stage — unlike pipeline_end_ts-start_ts which, in STREAMING
    # mode, is dominated by queue-wait and is NOT a per-video processing time.
    stage_compute: dict[str, float] = defaultdict(float)
    n_videos = 0
    if pv_dir.is_dir():
        for pv in pv_dir.glob("*.json"):
            try:
                d = json.loads(pv.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            n_videos += 1
            if isinstance(d.get("pipeline_start_ts"), (int, float)):
                starts.append(d["pipeline_start_ts"])
            if isinstance(d.get("pipeline_end_ts"), (int, float)):
                ends.append(d["pipeline_end_ts"])
            st = d.get("stage_timestamps", {})
            if isinstance(st, dict):
                stems = {k[: -len("_start")] for k in st if k.endswith("_start")}
                for stem in stems:
                    s, e = st.get(f"{stem}_start"), st.get(f"{stem}_end")
                    if isinstance(s, (int, float)) and isinstance(e, (int, float)) and e >= s:
                        stage_compute[stem] += e - s

    def _stage_total(needle: str) -> float:
        return round(sum(v for k, v in stage_compute.items() if needle in k.lower()), 1)

    caption_secs = _stage_total("caption")
    judge_secs = _stage_total("judge")
    # Reported with a caveat: inflated when the run was resumed across separate Slurm jobs.
    wall = (max(ends) - min(starts)) if (starts and ends) else None

    n_caps = sum(1 for r in recs if r.caption)
    est_tokens = sum(len(r.caption.split()) for r in recs if r.caption) * 1.3  # ~1.3 tok/word
    out.update(
        n_videos=n_videos or None,
        n_captions=n_caps,
        wall_clock_span_s=round(wall, 1) if wall else None,
        caption_compute_s=caption_secs or None,
        caption_compute_per_caption_s=round(caption_secs / n_caps, 3) if (caption_secs and n_caps) else None,
        est_caption_tokens_per_s=round(est_tokens / caption_secs, 1) if caption_secs else None,
        judge_compute_s=judge_secs or None,
    )
    if summary_path.exists():
        try:
            summ = json.loads(summary_path.read_text())
            if isinstance(summ, dict):
                out["summary_total_clips"] = summ.get("num_clips") or summ.get("total_clips")
        except (OSError, json.JSONDecodeError):
            pass
    return out


# ── Metric: GT-free sub-action completeness ────────────────────────────────────


def subaction_completeness(recs: list[WindowRec], fine_recs: list[WindowRec]) -> dict[str, Any]:
    """How completely a coarse-window caption covers the sub-actions in its span — no GT.

    Uses a *finer-grained* caption run of the same videos as the reference: action-aligned
    captions (``--gt-windows-source agibot``) or a smaller fixed window. For each coarse
    window we find every fine caption whose frame range falls inside it and measure how
    much of each fine sub-caption's content the coarse caption reproduces
    (``mention_recall``). A low score means the coarse caption omits sub-actions — the
    minor-action problem, detected with no ground truth at all.

    Reported:
      * ``subaction_completeness``     — mean coverage of sub-captions by their coarse caption.
      * ``n_windows_with_subactions``  — coarse windows that contained ≥2 fine sub-captions.
      * ``mean_subactions_per_window`` — average fine sub-captions per coarse window.
    """
    # index fine captions by source video name → list of (start, end, caption)
    fine_index: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for fr in fine_recs:
        if fr.caption:
            fine_index[Path(fr.source_video).name].append((fr.start_frame, fr.end_frame, fr.caption))

    completeness_vals: list[float] = []
    subaction_counts: list[int] = []
    n_multi = 0
    for r in recs:
        if not r.caption:
            continue
        subs = [
            cap
            for (s, e, cap) in fine_index.get(Path(r.source_video).name, [])
            if s >= r.start_frame and e <= r.end_frame and not (s == r.start_frame and e == r.end_frame)
        ]
        if not subs:
            continue
        subaction_counts.append(len(subs))
        if len(subs) >= 2:  # noqa: PLR2004
            n_multi += 1
        # coverage of each sub-caption's content by the coarse caption
        completeness_vals.append(statistics.mean(mention_recall(r.caption, sub) for sub in subs))

    if not completeness_vals:
        return {"subaction_completeness": None, "n_windows_with_subactions": 0}
    return {
        "subaction_completeness": round(statistics.mean(completeness_vals), 4),
        "n_windows_with_subactions": n_multi,
        "mean_subactions_per_window": round(statistics.mean(subaction_counts), 2),
    }


# ── Cross-model similarity (across runs) ───────────────────────────────────────


def cross_model_similarity(runs: dict[str, list[WindowRec]]) -> dict[str, Any]:
    """Bag-of-words cosine between caption models on the same (video, window)."""
    if len(runs) < 2:  # noqa: PLR2004
        return {}
    # index by (source_video_name, window_key)
    indexed: dict[str, dict[tuple[str, str], str]] = {}
    for name, recs in runs.items():
        indexed[name] = {(Path(r.source_video).name, r.window_key): r.caption for r in recs if r.caption}
    names = sorted(runs)
    out: dict[str, Any] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            keys = set(indexed[na]) & set(indexed[nb])
            if not keys:
                continue
            sims = [bow_cosine(indexed[na][k], indexed[nb][k]) for k in keys]
            out[f"{na}__vs__{nb}"] = {
                "n_shared_windows": len(keys),
                "mean_cosine": round(statistics.mean(sims), 4),
                "median_cosine": round(statistics.median(sims), 4),
            }
    return out


# ── Ranking ────────────────────────────────────────────────────────────────────


def _composite_quality(report: dict[str, Any]) -> float | None:
    """Combine available quality signals into one 0-1 score (higher = better).

    Components (each used only if present):
      * semantic F1 if available, else lexical action F1 (grounding correctness);
      * unique-caption ratio (guards against a model that wins by repeating one caption);
      * object continuity and (1 - contradiction rate).

    Notes on what is deliberately NOT rewarded: caption *length* (the semantic F1's precision
    term already penalises padding), and intra-clip consistency is dropped from the composite
    because a degenerate model that repeats itself would score high on it — uniqueness is used
    instead. Efficiency is reported separately and not folded in.
    """
    g = report.get("grounding", {})
    c = report.get("consistency", {})
    cap = report.get("captions", {})
    grounding_score = g.get("sem_f1") if g.get("sem_f1") is not None else g.get("action_f1")
    candidates = (
        grounding_score,
        cap.get("unique_caption_ratio"),
        c.get("object_continuity"),
        (1.0 - c["contradiction_rate"]) if isinstance(c.get("contradiction_rate"), (int, float)) else None,
    )
    parts: list[float] = [float(v) for v in candidates if isinstance(v, (int, float))]
    return round(statistics.mean(parts), 4) if parts else None


def build_report(
    runs: dict[str, list[WindowRec]],
    run_dirs: dict[str, Path],
    manual_gt: dict[str, bool] | None,
    coverage_threshold: float,
    fine_recs: list[WindowRec] | None = None,
    encoders: dict[str, Any] | None = None,  # {model_name: encode_callable}
) -> dict[str, Any]:
    """Assemble the full benchmark report + ranking."""
    encoders = encoders or {}
    # primary text encoder drives the table/composite/order columns; all encoders are
    # reported under grounding["semantic_models"] so several can be compared side by side.
    primary_encode = next(iter(encoders.values())) if encoders else None
    per_run: dict[str, Any] = {}
    for name, recs in runs.items():
        grounding = grounding_stats(recs, coverage_threshold)
        grounding.update(granularity_stats(recs))
        grounding.update(idle_stats(recs, coverage_threshold))
        grounding.update(order_stats(recs, primary_encode))
        grounding.update(robustness_stats(recs))
        if fine_recs:
            grounding.update(subaction_completeness(recs, fine_recs))
        if encoders:
            sem_by_model = {mname: semantic_stats(recs, enc) for mname, enc in encoders.items()}
            primary_name = next(iter(sem_by_model))
            grounding.update(sem_by_model[primary_name])  # flat sem_* from primary → table/composite
            grounding["semantic_models"] = sem_by_model  # full per-model for comparison
        per_run[name] = {
            "captions": caption_stats(recs),
            "judges": judge_stats(recs, manual_gt),
            "grounding": grounding,
            "consistency": consistency_stats(recs),
            "efficiency": efficiency_stats(run_dirs[name], recs),
        }
        per_run[name]["quality_score"] = _composite_quality(per_run[name])

    ranking = sorted(
        per_run,
        key=lambda n: (per_run[n]["quality_score"] is not None, per_run[n]["quality_score"] or 0.0),
        reverse=True,
    )
    return {
        "runs": per_run,
        "cross_model_similarity": cross_model_similarity(runs),
        "ranking": [{"run": n, "quality_score": per_run[n]["quality_score"]} for n in ranking],
    }


# ── Pretty printing ────────────────────────────────────────────────────────────


def _fmt(v: Any) -> str:  # noqa: ANN401 — formats arbitrary JSON metric values
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact ranking table to stdout."""
    cols: list[tuple[str, Any]] = [
        ("n_cap", lambda r: r["captions"].get("n_captions")),
        ("lex_f1", lambda r: r["grounding"].get("action_f1")),
        ("sem_f1", lambda r: r["grounding"].get("sem_f1")),
        ("sem_prec", lambda r: r["grounding"].get("sem_precision")),
        ("boundary", lambda r: r["grounding"].get("boundary_recall")),
        ("order", lambda r: r["grounding"].get("order_consistency_lexical")),
        ("granul", lambda r: r["grounding"].get("granularity_aware_recall")),
        ("idle_rec", lambda r: r["grounding"].get("idle_caption_recall")),
        ("uniq", lambda r: r["captions"].get("unique_caption_ratio")),
        ("words", lambda r: r["captions"].get("mean_words")),
        ("contra", lambda r: r["consistency"].get("contradiction_rate")),
        ("quality", lambda r: r.get("quality_score")),
    ]
    header = " | ".join([f"{'run':>13}", *(f"{label:>13}" for label, _ in cols)])
    print("\n" + "=" * len(header))
    print("CAPTION/JUDGE BENCHMARK — model ranking (higher quality = better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for entry in report["ranking"]:
        n = entry["run"]
        r = report["runs"][n]
        row = " | ".join([f"{n[:13]:>13}", *(f"{_fmt(fn(r)):>13}" for _, fn in cols)])
        print(row)
    print("=" * len(header))
    if report["cross_model_similarity"]:
        print("\nCross-model caption similarity (BoW cosine on shared windows):")
        for pair, d in report["cross_model_similarity"].items():
            print(f"  {pair}: mean={_fmt(d['mean_cosine'])} (n={d['n_shared_windows']})")
    # judge agreement, printed per run
    for n in report["runs"]:
        agr = report["runs"][n]["judges"].get("agreement", {})
        if agr:
            print(f"\nJudge agreement [{n}]:")
            for pair, d in agr.items():
                print(f"  {pair}: agree={_fmt(d['raw_agreement'])} kappa={_fmt(d['cohens_kappa'])} (n={d['n']})")
    # per-encoder semantic comparison (when several --semantic-model are given)
    for n in report["runs"]:
        sm = report["runs"][n]["grounding"].get("semantic_models")
        if sm and len(sm) > 1:
            print(f"\nSemantic text↔text encoders [{n}] (compare which fits best):")
            for mname, d in sm.items():
                print(
                    f"  {mname:>16}: sem_f1={_fmt(d.get('sem_f1'))} "
                    f"recall={_fmt(d.get('sem_recall'))} precision={_fmt(d.get('sem_precision'))} "
                    f"len_bias(f1)={_fmt((d.get('length_robustness') or {}).get('sem_f1_vs_length_r'))}"
                )
    # length-bias diagnostic: how much each grounding score correlates with caption length.
    # |r| near 0 = length-robust; high positive = a longer caption inflates the score.
    for n in report["runs"]:
        lr = report["runs"][n]["grounding"].get("length_robustness")
        if lr:
            print(f"\nLength-bias [{n}] (corr of score with caption length; ~0 = robust, high = gameable):")
            print(
                f"  lexical_recall: r={_fmt(lr.get('lexical_recall_vs_length_r'))}   "
                f"sem_recall: r={_fmt(lr.get('sem_recall_vs_length_r'))}   "
                f"sem_f1: r={_fmt(lr.get('sem_f1_vs_length_r'))}   "
                f"(mean {_fmt(lr.get('mean_caption_words'))} words)"
            )
    print()


# ── Manual GT loader ───────────────────────────────────────────────────────────


def load_manual_gt(path: Path) -> dict[str, bool]:
    """Load a manual annotation file → {key: is_correct} where key is clip|window or video|window.

    Accepts the existing ``annotation_*_ground_truth.json`` shapes:
      * ``{key: "correct"|"incorrect"}`` or ``{key: bool}``
      * nested ``{clip_uuid: {window_key: "correct"|...}}``
    """
    raw = json.loads(path.read_text())
    out: dict[str, bool] = {}

    def _coerce(v: Any) -> bool | None:  # noqa: ANN401 — coerces arbitrary JSON annotation values
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("correct", "true", "1", "yes"):
                return True
            if s in ("incorrect", "false", "0", "no"):
                return False
        return None

    for k, v in raw.items():
        if isinstance(v, dict):
            for wk, vv in v.items():
                c = _coerce(vv if not isinstance(vv, dict) else vv.get("label"))
                if c is not None:
                    out[f"{k}|{wk}"] = c
        else:
            c = _coerce(v)
            if c is not None:
                out[k] = c
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse args, build the report, print + optionally write it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Caption model name and its run output dir, e.g. qwen=/path/out. Repeatable.",
    )
    ap.add_argument("--manual-gt", type=Path, default=None, help="Optional manual GT annotation JSON for P/R/F1.")
    ap.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.5,
        help="Windows with GT action coverage below this are counted as 'uncovered' (default 0.5).",
    )
    ap.add_argument(
        "--fine-run",
        type=Path,
        default=None,
        metavar="DIR",
        help="Optional finer-grained / action-aligned caption run (e.g. --gt-windows-source output) "
        "used as a GT-free reference for sub-action completeness.",
    )
    ap.add_argument(
        "--semantic-model",
        action="append",
        default=None,
        metavar="[NAME=]DIR",
        help="sentence-transformers model dir for semantic grounding (sem_recall/precision/f1). "
        "Repeatable — pass several to compare encoders side by side, e.g. "
        "--semantic-model minilm=.../all-MiniLM-L6-v2 --semantic-model bge=.../bge-large-en-v1.5.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write the full JSON report here.")
    args = ap.parse_args()

    run_dirs: dict[str, Path] = {}
    for spec in args.run:
        if "=" not in spec:
            ap.error(f"--run must be NAME=DIR, got {spec!r}")
        name, _, path = spec.partition("=")
        run_dirs[name] = Path(path)

    runs: dict[str, list[WindowRec]] = {}
    for name, d in run_dirs.items():
        recs = load_run(d)
        runs[name] = recs
        print(f"[loaded] {name}: {len(recs)} windows from {d}")

    manual_gt = load_manual_gt(args.manual_gt) if args.manual_gt else None
    if manual_gt is not None:
        print(f"[loaded] manual GT: {len(manual_gt)} labelled windows")

    fine_recs = load_run(args.fine_run) if args.fine_run else None
    if fine_recs is not None:
        print(f"[loaded] fine-grained reference: {len(fine_recs)} windows from {args.fine_run}")

    encoders: dict[str, Any] = {}
    for spec in args.semantic_model or []:
        name, _, path = spec.partition("=") if "=" in spec else ("", "", spec)
        path_obj = Path(path)
        model_name = name or path_obj.name
        enc = build_semantic_encoder(path_obj)
        if enc is not None:
            encoders[model_name] = enc

    report = build_report(runs, run_dirs, manual_gt, args.coverage_threshold, fine_recs, encoders)
    print_summary(report)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[written] full report → {args.out}")


if __name__ == "__main__":
    main()
