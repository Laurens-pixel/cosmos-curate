# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end downstream_eval run over REAL pipeline output (no mock data).

Default dataset: ``agibot_qwen_evaluate_30vid_output`` — 30 AgiBotWorld episodes
(tasks 327/352/354), Qwen 2.5-VL captions, C-RADIO clip embeddings, ``gemma4_e4b``
in-pipeline judge verdicts. GT comes from ``agibot_alpha/task_info`` (per-episode
``action_config``: frame ranges + skill label + action text).

All paths are CLI flags so this same script runs against the full 2587-episode set
(``agibot_2587_bs16_output``) once its head_color.mp4 videos finish re-downloading —
just pass ``--output-dir .../agibot_2587_bs16_output --video-source-dir
.../agibot_head_2587``. That set has no C-RADIO embeddings (run with
``--no-generate-embeddings``), so action recognition there will fall back to
zero-shot-language only; CLIPScore/temporal-coherence/judge-aggregation/retrieval
all work the same regardless of embeddings.

Exercises every pillar except policy learning (handled separately via real LeRobot +
real LIBERO rollout, not this suite's synthetic-env smoke test):
    segmentation -> caption quality (reference + CLIPScore + temporal)
    -> judge aggregation -> retrieval -> action recognition

Run::

    python downstream_eval/run_real_agibot30.py
    python downstream_eval/run_real_agibot30.py --output-dir .../agibot_2587_bs16_output \\
        --video-source-dir .../agibot_head_2587
"""

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cosmos_curate.pipelines.video.evaluation.gt_sources.agibot import AgibotTaskInfoGt  # noqa: E402

from downstream_eval.captioning.clipscore import clipscore_from_embeddings  # noqa: E402
from downstream_eval.captioning.judge_aggregate import aggregate_judgments, iter_judge_records_from_clips  # noqa: E402
from downstream_eval.captioning.reference_metrics import bertscore, cider_d, meteor  # noqa: E402
from downstream_eval.captioning.temporal import narrative_coherence  # noqa: E402
from downstream_eval.common.io import load_clip_embeddings, load_clip_records  # noqa: E402
from downstream_eval.common.types import ClipRecord, Segment, WindowRecord  # noqa: E402
from downstream_eval.downstream.action_recognition_runner import (  # noqa: E402
    run_linear_probe,
    run_nearest_centroid,
    run_zero_shot_language,
)
from downstream_eval.downstream.encoders import build_text_encoder  # noqa: E402
from downstream_eval.downstream.retrieval_runner import RetrievalRun, run_caption_retrieval  # noqa: E402
from downstream_eval.segmentation.metrics import evaluate_segmentation  # noqa: E402

ROOT = pathlib.Path("/gpfs/work4/0/prjs0951/Sem")
DEFAULT_OUTPUT_DIR = ROOT / "cosmos_curate_local_workspace" / "agibot_qwen_evaluate_30vid_output"
DEFAULT_TASK_INFO_DIR = ROOT / "agibot_alpha" / "task_info"
DEFAULT_VIDEO_SOURCE_DIR = ROOT / "agibot_head_30vid_test"
DEFAULT_CLIP_MODEL_DIR = ROOT / "cosmos_curate_local_workspace" / "models" / "openai" / "clip-vit-base-patch32"

_VIDEO_RE = re.compile(r"^(\d+)_(\d+)_head_color\.mp4$")
_CLIPSCORE_FRAMES_PER_CLIP = 4


def _show(title: str, payload: object) -> None:
    import sys
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, default=lambda o: round(float(o), 4) if isinstance(o, float) else str(o)))
    sys.stdout.flush()


def clean_caption(raw: str) -> str:
    """Extract the plain caption string from Qwen's ``{"caption": ..., "action": ..., "object": ...}`` wrapper."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("caption"):
            return str(obj["caption"]).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return raw.strip()


def load_task_episode_actions(task_info_dir: pathlib.Path, task_ids: set[str]) -> dict[tuple[str, str], list[dict]]:
    """Load full per-episode ``action_config`` lists for the given task ids."""
    out: dict[tuple[str, str], list[dict]] = {}
    for task_id in task_ids:
        path = task_info_dir / f"task_{task_id}.json"
        if not path.exists():
            print(f"  [warn] missing task_info file: {path}")
            continue
        episodes = json.loads(path.read_text())
        for ep in episodes:
            ep_id = str(ep.get("episode_id", ""))
            actions = ep.get("label_info", {}).get("action_config", [])
            out[(task_id, ep_id)] = actions
    return out


class LocalClipEncoder:
    """Wraps the project's already-downloaded transformers CLIP model (no ``open_clip``,
    no extra network dependency / version-conflict risk -- see CLAUDE.md sentence-transformers
    incident). Lazily loaded so the rest of the script runs even if frames are unavailable."""

    def __init__(self, model_dir: pathlib.Path) -> None:
        self._model_dir = model_dir
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._model = CLIPModel.from_pretrained(str(self._model_dir)).eval()
        self._processor = CLIPProcessor.from_pretrained(str(self._model_dir))

    def encode(self, frames: list, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Encode parallel (frame, text) pairs -> (image_embeds, text_embeds), both (N, D)."""
        self._ensure_loaded()
        inputs = self._processor(text=texts, images=frames, return_tensors="pt", padding=True)
        with self._torch.no_grad():
            out = self._model(**inputs)
        return out.image_embeds.numpy(), out.text_embeds.numpy()


def extract_clip_frames(
    video_source_dir: pathlib.Path, clip: ClipRecord, window_start_s: float, window_end_s: float, n_frames: int
) -> list:
    """Decode ``n_frames`` evenly-spaced RGB frames from the window's time range in the
    SOURCE video (clips/ is not persisted by this pipeline run). Returns [] if the video
    is unavailable (e.g. broken symlink while agibot_alpha is being re-downloaded)."""
    video_path = video_source_dir / pathlib.Path(clip.source_video).name
    if not video_path.exists():
        return []
    import av
    from PIL import Image

    try:
        container = av.open(str(video_path))
    except (OSError, av.AVError):
        return []
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    target_times = np.linspace(window_start_s, max(window_start_s, window_end_s - 1.0 / fps), n_frames)
    frames: list = []
    next_idx = 0
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base) if frame.pts is not None else frame.time
        if next_idx < len(target_times) and t >= target_times[next_idx]:
            frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
            next_idx += 1
            if next_idx >= len(target_times):
                break
    container.close()
    return frames


def _retrieval_breakdown(
    run: RetrievalRun,
    clip_to_task: dict[str, str],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, object]:
    """Within-task and cross-task retrieval breakdown using the already-computed similarity matrix.

    Within-task MRR: given a GT label, rank its caption against only other captions from the
    same task. Pool size = number of clips in that task. Tests fine-grained discrimination.

    Cross-task accuracy (task@K): given a GT label, what fraction of the top-K retrieved clips
    (global gallery) come from the same task? Tests coarse task-level discrimination.
    """
    sim = run.similarity           # (Q, G)
    query_ids = run.query_ids
    gallery_ids = run.gallery_ids

    gallery_task = np.array([clip_to_task.get(uid, "") for uid in gallery_ids])
    gallery_id_to_idx = {uid: j for j, uid in enumerate(gallery_ids)}

    within_mrr_all: list[float] = []
    within_recall: dict[int, list[float]] = {k: [] for k in ks}
    within_mrr_by_task: dict[str, list[float]] = defaultdict(list)
    task_at_k: dict[int, list[float]] = {k: [] for k in ks}

    for q_idx, q_uid in enumerate(query_ids):
        q_task = clip_to_task.get(q_uid, "")
        correct_g_idx = gallery_id_to_idx.get(q_uid, -1)
        if correct_g_idx < 0:
            continue

        q_sims = sim[q_idx]

        # ── within-task: restrict gallery to same task ────────────────────
        same_task_indices = np.where(gallery_task == q_task)[0]
        correct_local = np.where(same_task_indices == correct_g_idx)[0]
        if len(correct_local) == 0:
            continue
        same_sims = q_sims[same_task_indices]
        correct_sim = same_sims[correct_local[0]]
        within_rank = int(np.sum(same_sims > correct_sim))
        rr = 1.0 / (within_rank + 1)
        within_mrr_all.append(rr)
        within_mrr_by_task[q_task].append(rr)
        for k in ks:
            within_recall[k].append(1.0 if within_rank < k else 0.0)

        # ── cross-task accuracy: top-K task composition (global gallery) ──
        full_ranking = np.argsort(-q_sims)
        for k in ks:
            top_k_tasks = gallery_task[full_ranking[:k]]
            task_at_k[k].append(float(np.mean(top_k_tasks == q_task)))

    task_pool_sizes = {t: int(np.sum(gallery_task == t)) for t in sorted(within_mrr_by_task)}

    return {
        "within_task": {
            "mrr": round(float(np.mean(within_mrr_all)), 4) if within_mrr_all else 0.0,
            **{f"recall@{k}": round(float(np.mean(within_recall[k])), 4) for k in ks},
            "per_task_mrr": {t: round(float(np.mean(v)), 4) for t, v in sorted(within_mrr_by_task.items())},
            "pool_sizes": task_pool_sizes,
        },
        "cross_task_accuracy": {
            f"task@{k}": round(float(np.mean(task_at_k[k])), 4) for k in ks
        },
        "note": (
            "within_task MRR: rank correct caption among same-task clips only. "
            "cross_task task@K: fraction of top-K retrieved clips sharing the query's task."
        ),
    }


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-info-dir", type=pathlib.Path, default=DEFAULT_TASK_INFO_DIR)
    parser.add_argument("--video-source-dir", type=pathlib.Path, default=DEFAULT_VIDEO_SOURCE_DIR)
    parser.add_argument("--clip-model-dir", type=pathlib.Path, default=DEFAULT_CLIP_MODEL_DIR)
    parser.add_argument("--embed-algorithm", default="cradio")
    parser.add_argument("--skip-clipscore", action="store_true", help="Skip frame decode + CLIPScore section")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip slow BERTScore computation")
    args = parser.parse_args()

    print(f"Loading clips from {args.output_dir} ...")
    clips: list[ClipRecord] = load_clip_records(args.output_dir)
    embeddings = load_clip_embeddings(args.output_dir, algorithm=args.embed_algorithm)
    print(f"  {len(clips)} clips, {len(embeddings)} embeddings")

    parsed = [(c, _VIDEO_RE.match(pathlib.Path(c.source_video).name)) for c in clips]
    task_ids = {m.group(1) for _, m in parsed if m}
    print(f"  task ids present: {sorted(task_ids)}")

    episode_actions = load_task_episode_actions(args.task_info_dir, task_ids)
    gt_lookup = AgibotTaskInfoGt(args.task_info_dir)

    gt_segments_by_video: dict[str, list[Segment]] = {}
    candidates: list[str] = []
    references: list[list[str]] = []
    skill_labels: list[str] = []
    clip_uuid_for_label: list[str] = []
    clip_records_for_retrieval: list[ClipRecord] = []
    retrieval_refs: dict[str, str] = {}
    per_video_windows: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    clipscore_items: list[tuple[ClipRecord, float, float, str]] = []  # (clip, w_start_s, w_end_s, caption)

    for clip, m in parsed:
        if m is None:
            continue
        task_id, episode_id = m.group(1), m.group(2)
        fps = clip.framerate or 30.0
        actions = episode_actions.get((task_id, episode_id), [])

        if clip.source_video not in gt_segments_by_video:
            gt_segments_by_video[clip.source_video] = [
                Segment(start=a["start_frame"] / fps, end=a["end_frame"] / fps, label=a.get("skill"))
                for a in actions
            ]

        primary_done = False
        for window in clip.windows:
            raw_caption = window.captions.get("qwen", "")
            if not raw_caption:
                continue
            caption = clean_caption(raw_caption)
            abs_start = round(clip.span[0] * fps) + window.start_frame
            abs_end = round(clip.span[0] * fps) + window.end_frame
            gt_text, extras = gt_lookup.lookup(clip.source_video, abs_start, abs_end)

            w_start_s = clip.span[0] + window.start_frame / fps
            w_end_s = clip.span[0] + window.end_frame / fps
            per_video_windows[clip.source_video].append((w_start_s, w_end_s, caption))

            if not gt_text:
                continue
            if not primary_done:
                candidates.append(caption)
                references.append([gt_text])
                skill_labels.append(str(extras.get("gt_skill", "")))
                clip_uuid_for_label.append(clip.uuid)
                clip_records_for_retrieval.append(
                    ClipRecord(uuid=clip.uuid, source_video=clip.source_video, span=clip.span,
                               windows=[WindowRecord(window.start_frame, window.end_frame, {"qwen": caption})])
                )
                retrieval_refs[clip.uuid] = gt_text
                clipscore_items.append((clip, w_start_s, w_end_s, caption))
                primary_done = True

    print(f"  matched {len(candidates)} clips with GT action text (out of {len(clips)})")

    # ---- 1. Segmentation quality ----
    seg_report: dict[str, dict[str, float]] = {}
    for video, gt_segments in gt_segments_by_video.items():
        pred_segments = [c.to_segment() for c, m in parsed if m and c.source_video == video]
        if not gt_segments or not pred_segments:
            continue
        seg_report[pathlib.Path(video).name] = evaluate_segmentation(pred_segments, gt_segments, tolerance=1.0)
    if seg_report:
        agg = {
            key: float(np.nanmean([v[key] for v in seg_report.values()]))
            for key in next(iter(seg_report.values()))
        }
        _show("Segmentation quality (per-video)", seg_report)
        _show("Segmentation quality (mean over videos)", agg)

    # ---- 2. Caption quality: reference-based (real bertscore + nltk meteor) ----
    cider = cider_d(candidates, references)
    met = meteor(candidates, references)
    bert = bertscore(candidates, references) if not args.skip_bertscore else None
    _show("Caption reference metrics (vs GT action_text)", {"cider_d": cider, "meteor": met, "bertscore": bert})

    # ---- 3. Caption quality: CLIPScore (real frames decoded from source video) ----
    if args.skip_clipscore:
        print("\n=== CLIPScore ===\nSkipped (--skip-clipscore).")
    else:
        encoder = LocalClipEncoder(args.clip_model_dir)
        all_image_embeds, all_text_embeds = [], []
        n_no_video = 0
        for clip, w_start_s, w_end_s, caption in clipscore_items:
            frames = extract_clip_frames(args.video_source_dir, clip, w_start_s, w_end_s, _CLIPSCORE_FRAMES_PER_CLIP)
            if not frames:
                n_no_video += 1
                continue
            img_emb, txt_emb = encoder.encode(frames, [caption] * len(frames))
            all_image_embeds.append(img_emb.mean(axis=0))
            all_text_embeds.append(txt_emb[0])
        if all_image_embeds:
            clipscore = clipscore_from_embeddings(np.stack(all_image_embeds), np.stack(all_text_embeds))
            clipscore["videos_unavailable"] = n_no_video
            _show("CLIPScore (real frames, local CLIP)", clipscore)
        else:
            print(f"\n=== CLIPScore ===\nSkipped: 0/{len(clipscore_items)} source videos available "
                  f"under {args.video_source_dir} (re-download pending?).")

    # ---- 4. Caption quality: temporal coherence ----
    temporal_report: dict[str, object] = {}
    for video, windows in per_video_windows.items():
        windows_sorted = sorted(windows, key=lambda w: w[0])
        captions_only = [w[2] for w in windows_sorted]
        temporal_report[pathlib.Path(video).name] = {
            "num_windows": len(windows_sorted),
            "coherence": narrative_coherence(captions_only),
        }
    _show("Caption temporal coherence (per-video, lexical-fallback)", temporal_report)

    # ---- 5. Judge aggregation ----
    judge_summary = aggregate_judgments(iter_judge_records_from_clips(clips))
    _show("Judge aggregation (in-pipeline judge verdicts)", judge_summary)

    # ---- 6. Downstream: retrieval ----
    encoder = build_text_encoder("auto")
    print(f"\n[retrieval/recognition use text encoder: {encoder.name}]")
    retrieval_run = run_caption_retrieval(clip_records_for_retrieval, retrieval_refs, encoder=encoder, ks=(1, 5, 10))
    _show("Retrieval (GT action_text -> Qwen caption)", retrieval_run.metrics)

    clip_to_task = {clip.uuid: m.group(1) for clip, m in parsed if m}
    breakdown = _retrieval_breakdown(retrieval_run, clip_to_task, ks=(1, 5, 10))
    _show("Retrieval breakdown (within-task vs cross-task)", breakdown)

    # ---- 7. Downstream: action recognition (skill label) ----
    class_names = sorted({s for s in skill_labels if s})
    if len(class_names) >= 2:  # noqa: PLR2004
        labels = np.array([class_names.index(s) for s in skill_labels], dtype=np.int64)
        kept_uuids = [uid for uid in clip_uuid_for_label if uid in embeddings]
        print(f"\n[action recognition] classes={class_names} label_counts={np.bincount(labels).tolist()} "
              f"embeddings_available={len(kept_uuids)}/{len(clip_uuid_for_label)}")

        results: dict[str, dict[str, float]] = {}
        if len(kept_uuids) >= 4:  # noqa: PLR2004 - need at least a couple samples per split
            emb_mat = np.stack([embeddings[uid] for uid in kept_uuids])
            emb_labels = np.array(
                [labels[i] for i, uid in enumerate(clip_uuid_for_label) if uid in embeddings], dtype=np.int64
            )
            try:
                results["nearest_centroid"] = run_nearest_centroid(
                    emb_mat, emb_labels, class_names, train_frac=0.6, seed=1, ks=(1,)
                ).metrics
                results["linear_probe"] = run_linear_probe(
                    emb_mat, emb_labels, class_names, train_frac=0.6, seed=1, ks=(1,)
                ).metrics
            except (ValueError, RuntimeError) as exc:
                print(f"  [warn] embedding-based recognition skipped: {exc}")
        else:
            print(f"  [info] no/too-few clip embeddings ({args.embed_algorithm}) -- "
                  "zero-shot-language is the only recognition method available")
        results["zero_shot_language"] = run_zero_shot_language(
            candidates, labels, class_names, encoder=encoder, ks=(1,)
        ).metrics
        _show("Action recognition (skill label)", results)
    else:
        print("\n[action recognition] fewer than 2 distinct skill classes matched — skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
