# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for the evaluation suite.

Builds a mock cosmos-curate output tree, loads it through the public loaders, runs every
evaluator (segmentation, captioning, downstream), prints a readable report, and asserts a
few invariants. Run directly::

    python -m downstream_eval.run_smoke

Heavy optional metrics (BERTScore, model-based CLIPScore) are skipped gracefully when their
dependencies are absent.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from downstream_eval.captioning.judge_aggregate import (
    aggregate_judgments,
    iter_judge_records,
    iter_judge_records_from_clips,
)
from downstream_eval.captioning.clipscore import clipscore_from_embeddings
from downstream_eval.captioning.reference_metrics import bertscore, cider_d, meteor
from downstream_eval.captioning.temporal import (
    action_order_consistency,
    narrative_coherence,
    temporal_grounding_accuracy,
)
from downstream_eval.common.io import (
    load_clip_embeddings,
    load_clip_records,
    load_window_captions,
    load_window_judgments,
    predicted_segments_by_video,
)
from downstream_eval.downstream.action_recognition_runner import (
    run_linear_probe,
    run_nearest_centroid,
    run_zero_shot_language,
)
from downstream_eval.downstream.encoders import build_text_encoder
from downstream_eval.downstream.policy_learning.synthetic import (
    NoisyPolicy,
    ScriptedExpert,
    run_synthetic_rollouts,
)
from downstream_eval.downstream.retrieval_runner import run_caption_retrieval
from downstream_eval.downstream.robot_completion import robot_task_metrics
from downstream_eval.mock_data import build_mock_dataset
from downstream_eval.segmentation.metrics import evaluate_segmentation


def _show(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, default=lambda o: round(float(o), 4) if isinstance(o, float) else str(o)))


def run(root: Path) -> None:  # noqa: PLR0915
    gt = build_mock_dataset(root, seed=7)

    clips = load_clip_records(gt.output_dir)
    captions = load_window_captions(gt.output_dir)
    judgments = load_window_judgments(gt.output_dir)
    embeddings = load_clip_embeddings(gt.output_dir, algorithm="cradio")

    assert len(clips) == len(gt.clips), "clip records did not round-trip"
    assert captions, "captions file missing"
    assert len(embeddings) == len(gt.clips), "embeddings did not round-trip"

    # ---- Segmentation ----
    pred_by_video = predicted_segments_by_video(clips)
    seg_report: dict[str, dict[str, float]] = {}
    for video, gt_segments in gt.gt_segments.items():
        result = evaluate_segmentation(pred_by_video[video], gt_segments, tolerance=0.5)
        seg_report[video] = result
        assert 0.0 <= result["boundary_f1"] <= 1.0
        assert result["fragmentation_index"] >= 1.0
    _show("Segmentation quality", seg_report)

    # ---- Captioning: reference-based ----
    candidates = [c.caption for c in gt.clips]
    references = [c.references for c in gt.clips]
    cider = cider_d(candidates, references)
    met = meteor(candidates, references)
    bert = bertscore(candidates, references)
    assert cider["cider_d"] > 0.0
    assert 0.0 <= met["meteor"] <= 1.0
    _show("Caption reference metrics", {"cider": cider, "meteor": met, "bertscore": bert})

    # ---- Captioning: CLIPScore (math path, synthetic aligned text embeddings) ----
    clip_mat = np.stack([embeddings[c.uuid] for c in gt.clips])
    text_mat = clip_mat + 0.05 * np.random.default_rng(1).standard_normal(clip_mat.shape).astype(np.float32)
    clipscore = clipscore_from_embeddings(clip_mat, text_mat)
    assert clipscore["clipscore"] > 0.0
    _show("CLIPScore (from embeddings)", clipscore)

    # ---- Captioning: temporal coherence ----
    temporal_report: dict[str, object] = {}
    for video, intervals in gt.gt_action_intervals.items():
        video_clips = [c for c in gt.clips if c.source_video == video]
        windows = [(c.start, c.end, c.caption) for c in video_clips]
        coherence = narrative_coherence([c.caption for c in video_clips])
        grounding = temporal_grounding_accuracy(windows, intervals)
        pred_labels = [c.label for c in video_clips]
        gt_labels = [lbl for _, _, lbl in intervals]
        order = action_order_consistency(pred_labels, gt_labels)
        temporal_report[video] = {"coherence": coherence, "grounding": grounding, "order": order}
        assert 0.0 <= grounding["temporal_grounding_accuracy"] <= 1.0
    _show("Caption temporal coherence", temporal_report)

    # ---- Captioning: judge aggregation ----
    agg_from_file = aggregate_judgments(iter_judge_records(judgments))
    agg_from_clips = aggregate_judgments(iter_judge_records_from_clips(clips))
    assert agg_from_file.keys() == agg_from_clips.keys()
    _show("Judge aggregation", agg_from_file)

    # ---- Downstream: retrieval (actually run caption retrieval, GT refs -> generated captions) ----
    encoder = build_text_encoder("auto")
    print(f"\n[retrieval/recognition use text encoder: {encoder.name}]")
    references = {c.uuid: c.label for c in gt.clips}  # GT instruction per clip
    retrieval_run = run_caption_retrieval(clips, references, encoder=encoder, ks=(1, 5))
    assert retrieval_run.metrics["recall@5"] >= 0.5, "discriminative captions should retrieve well"
    _show("Retrieval (caption retrieval task)", retrieval_run.metrics)
    print(f"example: query {retrieval_run.query_ids[0][:8]} -> top5 {[g[:8] for g in retrieval_run.top_k(0)]}")

    # ---- Downstream: action recognition (actually classify clips three ways) ----
    class_names = gt.class_names
    labels = np.array([class_names.index(c.label) for c in gt.clips], dtype=np.int64)
    clip_uuids = [c.uuid for c in gt.clips]
    emb_mat = np.stack([embeddings[uid] for uid in clip_uuids])

    centroid_run = run_nearest_centroid(emb_mat, labels, class_names, train_frac=0.6, seed=1, ks=(1, 5))
    probe_run = run_linear_probe(emb_mat, labels, class_names, train_frac=0.6, seed=1, ks=(1, 5))
    caption_texts = [c.caption for c in gt.clips]
    zeroshot_run = run_zero_shot_language(caption_texts, labels, class_names, encoder=encoder, ks=(1, 5))
    assert centroid_run.metrics["top1_accuracy"] >= 0.5
    assert zeroshot_run.metrics["top1_accuracy"] >= 0.5
    _show(
        "Action recognition (performed)",
        {
            "nearest_centroid": centroid_run.metrics,
            "linear_probe": probe_run.metrics,
            "zero_shot_language": zeroshot_run.metrics,
        },
    )

    # ---- Downstream: policy learning (actually roll out policies in synthetic env, then score) ----
    expert_rollouts = run_synthetic_rollouts(lambda _rng: ScriptedExpert(), num_episodes=12, seed=2)
    learned_rollouts = run_synthetic_rollouts(lambda rng: NoisyPolicy(noise=0.9, rng=rng), num_episodes=12, seed=2)
    expert_metrics = robot_task_metrics(expert_rollouts)
    learned_metrics = robot_task_metrics(learned_rollouts)
    assert math.isclose(expert_metrics["task_success_rate"], 1.0), "scripted expert should always succeed"
    assert expert_metrics["efficiency_ratio"] <= 1.0
    assert learned_metrics["task_success_rate"] <= expert_metrics["task_success_rate"]
    _show(
        "Policy learning (rollout -> task completion)",
        {"scripted_expert": expert_metrics, "noisy_learned_policy": learned_metrics},
    )

    print("\nAll smoke checks passed.")


def main() -> None:
    """Run the smoke test in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(tmp))


if __name__ == "__main__":
    main()
