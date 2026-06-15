<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `downstream_eval` — Evaluation suite for cosmos-curate outputs

This suite measures the **quality and usefulness** of the curation pipeline's outputs
(clips + captions + embeddings). It is intentionally decoupled from the pipeline: it reads
the on-disk output layout directly and **does not train any models**. It reuses the
artifacts the pipeline already produces — clip spans, captions, the in-pipeline VLM judge,
and C-RADIO/embedding vectors.

```
downstream_eval/
  common/            # data structures + loaders for the cosmos-curate output layout
  segmentation/      # 1. Segmentation Quality  (boundary, completeness, ordering)
  captioning/        # 2. Caption Quality       (reference, model-based, temporal, judge)
  downstream/        # 3. Downstream Task Perf.  (retrieval, action recognition, robot)
  mock_data.py       # generates a tiny pipeline-style output tree + ground truth
  run_smoke.py       # end-to-end smoke test over the mock data
  tests/             # pytest unit tests with hand-computed expectations
```

## Quick start

```bash
# End-to-end smoke test (no GPU / heavy deps required):
python -m downstream_eval.run_smoke

# Unit tests (the main `pytest` run ignores this dir via testpaths=tests):
pytest downstream_eval/tests -o addopts="" -p no:cacheprovider
```

Metric *math* depends only on `numpy`. Model-based metrics degrade gracefully when their
optional dependencies are absent:

| Metric | Optional dependency | Behaviour if missing |
| --- | --- | --- |
| BERTScore | `bert_score` | returns `{"available": false, ...}` |
| CLIPScore (from frames) | `open_clip` + `torch` | use `clipscore_from_embeddings` instead |
| METEOR (full) | `nltk` + wordnet | falls back to exact-match METEOR |
| Narrative coherence | any LLM judge callable | falls back to lexical continuity |

---

## Step 1 — Which downstream tasks are actually relevant?

Relevance is gated by **what ground truth each dataset actually has**. A metric is only
meaningful where the corresponding labels exist.

| Dataset | Has sub-task boundaries? | Has reference captions? | Has class labels? | Has actions + closed-loop sim? |
| --- | --- | --- | --- | --- |
| **LIBERO** | weak (single-task demos) | instruction (1 per demo) | task type | ✅ sim (closed-loop) |
| **DROID** | ❌ | instruction | task type | ⚠️ real robot only |
| **AgiBotWorld** | ✅ (`action_config` frames) | action text | skill/action | ⚠️ real robot only |
| **YouCook2** | ✅ (segment seconds) | sentence per step | recipe-step | ❌ |
| **InHARD** | ✅ (Anvil timestamps, online) | class label | action class ★ | ❌ |
| **nuScenes** | ❌ (scene-level only) | scene tags | scene tags (multi-label) | ❌ |

### 1. Segmentation Quality — *the core claim of the SBD work*
Only meaningful where temporally-localized sub-task GT exists.

- **Run on: AgiBotWorld, InHARD (online), YouCook2.**
- Skip on: LIBERO (each demo is a single task → no internal boundaries), DROID (no
  sub-task annotation), nuScenes (no temporal subdivisions within a scene).
- All three boundary metrics (Boundary F1, tIoU, MAE) and completeness metrics
  (over/under-segmentation, fragmentation) directly quantify the "robot demos lack obvious
  cuts → over/under-segment" failure mode that motivated replacing TransNetV2.
- Temporal ordering (Kendall's τ, Damerau-Levenshtein) needs *labelled* segments, so it is
  most informative on AgiBot/InHARD (action identities), less so on YouCook2.

### 2. Caption Quality — *applies broadly*
- **CLIPScore** is reference-free → run on **all six** datasets (robust default).
- **METEOR / CIDEr** need sentence-like references → best on **YouCook2, DROID, AgiBot,
  LIBERO**; report-only on InHARD/nuScenes (label/tag GT caps the achievable score).
- **BERTScore** (semantic) works wherever text GT exists.
- **VLM-as-judge** already runs in-pipeline for AgiBot/YouCook2/nuScenes/InHARD; we only
  *aggregate* it here.
- **Temporal coherence** (narrative, grounding, action-order) is meaningful for multi-step
  videos (**YouCook2, AgiBot, InHARD**); near-degenerate for single-action clips.

### 3. Downstream Task Performance
- **Retrieval (R@K, MRR, nDCG)** — **run on all datasets.** It only needs captions + clip
  embeddings (both already produced) and is the cheapest, broadest proxy for whether
  captions are discriminative. Highest value-per-effort.
- **Action Recognition / Classification** — needs class labels → **InHARD ★** (well-defined
  action classes), **AgiBotWorld** (skills); **nuScenes** as multi-label scene tags.
  Marginal for LIBERO/DROID/YouCook2.
- **Robot Task Completion** — the gold standard for "useful for policy learning", but it
  requires a policy + rollout environment. **LIBERO ★** (deterministic sim, closed-loop);
  AgiBot/DROID need real-robot or a learned simulator. This suite scores *rollout results*
  fed in from an external harness (see `downstream/README.md`); it does **not** train or
  roll out policies.

### Recommended evaluation ladder (cheap → expensive)
1. **Intrinsic** — segmentation + caption metrics (CLIPScore/CIDEr/judge). Run everywhere.
2. **Grounding / Retrieval** — retrieval + action recognition. Strong, cheap usefulness signal.
3. **Closed-loop policy** — LIBERO task completion. Heaviest; reserve for the headline result.

---

## Step 3 — Why no external repo was vendored

Per the task, an external repo is only cloned if it is **directly compatible** with the
cosmos-curate I/O format (`metas/v0/*.json`, `all_window_*.json`, `*_embd/*.pickle`). None
are: COCO-caption / `pycocoevalcap`, `bert_score`, CLIPScore, action-segmentation toolkits,
and LIBERO/LeRobot all assume their own formats. So:

- **Segmentation, retrieval, action recognition, CIDEr-D, METEOR** — written here in pure
  numpy/python (compact, dependency-light, unit-tested).
- **BERTScore / CLIPScore** — thin optional wrappers over the established libraries (no point
  reimplementing model inference) with graceful fallbacks.
- **Robot task completion** — a stable `EpisodeResult` schema + `RolloutProvider` protocol +
  JSON loader so LIBERO/LeRobot rollouts can be scored, without vendoring those heavy stacks.
