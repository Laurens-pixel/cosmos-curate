# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Shot-boundary / temporal-segmentation evaluation.

Scores a detector's *output* (the clip boundaries it produced) against ground-truth action
segments. It is **detector-agnostic** — it reads the clip spans any splitting method wrote to
``metas/v0/*.json`` (``duration_span``), so the same harness evaluates TransNetV2,
PySceneDetect, the semantic encoders (CLIP/SigLIP2/DINOv2/V-JEPA2) and the recursive VLM
methods. No model weights are needed here: this measures segmentation quality, not captions.

Metrics (the standard shot-boundary + action-segmentation suite):

* **Boundary P/R/F1 @ tolerance** — predicted cut points matched 1-to-1 to GT boundaries
  within ±tolerance seconds (classic TRECVID-style SBD metric). Reported at several
  tolerances (default ±0.5 s, ±1.0 s).
* **Segment F1 @ tIoU** — a predicted segment is a true positive if its temporal IoU with a
  GT segment exceeds the threshold (greedy 1-to-1). Reported at IoU ∈ {0.1, 0.25, 0.5} — the
  MS-TCN / action-segmentation standard.
* **Boundary distance** — mean seconds from each GT boundary to its nearest predicted
  boundary, and the reverse (localisation error, independent of any threshold).
* **Over-segmentation ratio** — #predicted / #GT segments (>1 = too many cuts, <1 = too few).
* **Coverage** — fraction of GT segments matched at IoU ≥ 0.5.

Usage::

    python -m cosmos_curate.pipelines.video.evaluation.benchmark_boundaries \\
        --run siglip2=/path/agibot_siglip2_output \\
        --run transnetv2=/path/agibot_transnetv2_output \\
        --gt-source agibot --gt-task-info /path/agibot_alpha/task_info \\
        --out boundary_report.json

Each ``--run NAME=DIR`` is one detector; they are ranked against each other. Pure stdlib —
runs anywhere, no GPU.
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

_EPS = 1e-3
_AGIBOT_NAME_RE = re.compile(r"^(\d+)_(\d+)_")

Segment = tuple[float, float]


# ── Loading predicted segments (detector output) ───────────────────────────────


def load_predicted_segments(run_dir: Path) -> tuple[dict[str, list[Segment]], dict[str, float]]:
    """Read per-video clip spans + source framerate from ``metas/v0/*.json``.

    Returns ``({video_name: [(start_s, end_s), ...]}, {video_name: fps})``.
    """
    segments: dict[str, list[Segment]] = {}
    fps: dict[str, float] = {}
    meta_dir = run_dir / "metas" / "v0"
    if not meta_dir.is_dir():
        return segments, fps
    for p in meta_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        src = Path(d.get("source_video", "")).name
        span = d.get("duration_span")
        if not src or not span:
            continue
        segments.setdefault(src, []).append((float(span[0]), float(span[1])))
        fr = d.get("framerate_source") or d.get("framerate")
        if fr:
            fps[src] = float(fr)
    return segments, fps


# ── Loading GT segments ─────────────────────────────────────────────────────────


def load_gt_segments_agibot(task_info_dir: Path, fps_lookup: dict[str, float]) -> dict[str, list[Segment]]:
    """Build ``{video_name: [(start_s, end_s), ...]}`` from AgiBot task_info action_config.

    GT frames are absolute source frames; converted to seconds with the per-video source fps
    (from the run's metas), defaulting to 30 fps when unknown.
    """
    # Cache action_config per (task_id, episode_id)
    by_episode: dict[tuple[str, str], list[Segment]] = {}
    for task_file in Path(task_info_dir).glob("task_*.json"):
        try:
            episodes = json.loads(task_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        task_id = task_file.stem.split("_")[-1]
        for ep in episodes:
            ep_id = str(ep.get("episode_id", ""))
            actions = ep.get("label_info", {}).get("action_config", [])
            segs = [(int(a.get("start_frame", 0)), int(a.get("end_frame", 0))) for a in actions]
            by_episode[(task_id, ep_id)] = segs  # type: ignore[assignment]

    out: dict[str, list[Segment]] = {}
    for video_name, fps in fps_lookup.items():
        m = _AGIBOT_NAME_RE.match(video_name)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        frame_segs = by_episode.get(key)
        if not frame_segs:
            continue
        f = fps or 30.0
        out[video_name] = [(s / f, e / f) for s, e in frame_segs if e > s]
    return out


# ── Geometry helpers ────────────────────────────────────────────────────────────


def internal_boundaries(segments: list[Segment]) -> list[float]:
    """Cut points strictly inside a video = segment starts/ends excluding the global span ends."""
    if not segments:
        return []
    segs = sorted(segments)
    v_start, v_end = segs[0][0], max(e for _, e in segs)
    bounds: set[float] = set()
    for s, e in segs:
        if s > v_start + _EPS:
            bounds.add(round(s, 3))
        if e < v_end - _EPS:
            bounds.add(round(e, 3))
    return sorted(bounds)


def temporal_iou(a: Segment, b: Segment) -> float:
    """Temporal IoU of two [start, end] intervals."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


# ── Core metrics ────────────────────────────────────────────────────────────────


def boundary_prf(pred: list[float], gt: list[float], tol: float) -> tuple[float, float, float, int, int, int]:
    """Greedy 1-to-1 boundary matching within ±tol → (precision, recall, f1, tp, fp, fn)."""
    used: set[int] = set()
    tp = 0
    for g in gt:
        best_i, best_d = None, tol + _EPS
        for i, p in enumerate(pred):
            if i in used:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    # No GT boundaries (single-action video): correct iff no spurious cuts predicted.
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def segment_f1_at_iou(pred: list[Segment], gt: list[Segment], thr: float) -> tuple[float, int]:
    """Greedy IoU-matched segment F1 at threshold ``thr`` → (f1, n_matched_gt)."""
    pairs = sorted(
        ((temporal_iou(ps, gs), i, j) for i, ps in enumerate(pred) for j, gs in enumerate(gt)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, tp


def _nearest_distances(src: list[float], dst: list[float]) -> list[float]:
    """For each boundary in ``src``, distance to the nearest boundary in ``dst`` (seconds)."""
    if not dst:
        return []
    return [min(abs(s - d) for d in dst) for s in src]


# ── Aggregation over a run ──────────────────────────────────────────────────────


def evaluate_run(
    pred_segs: dict[str, list[Segment]],
    gt_segs: dict[str, list[Segment]],
    tolerances: list[float],
    iou_thresholds: list[float],
) -> dict[str, Any]:
    """Aggregate boundary + segment metrics over all videos shared by pred and GT."""
    videos = sorted(set(pred_segs) & set(gt_segs))
    if not videos:
        return {"n_videos": 0, "note": "no videos shared between detector output and GT"}

    b_prf: dict[float, list[tuple[float, float, float]]] = {t: [] for t in tolerances}
    s_f1: dict[float, list[float]] = {t: [] for t in iou_thresholds}
    over_seg: list[float] = []
    gt_to_pred: list[float] = []
    pred_to_gt: list[float] = []
    coverage: list[float] = []
    n_pred_total = n_gt_total = 0

    for v in videos:
        psegs, gsegs = pred_segs[v], gt_segs[v]
        n_pred_total += len(psegs)
        n_gt_total += len(gsegs)
        pb, gb = internal_boundaries(psegs), internal_boundaries(gsegs)
        for t in tolerances:
            p, r, f1, *_ = boundary_prf(pb, gb, t)
            b_prf[t].append((p, r, f1))
        for thr in iou_thresholds:
            f1, _ = segment_f1_at_iou(psegs, gsegs, thr)
            s_f1[thr].append(f1)
        over_seg.append(len(psegs) / len(gsegs) if gsegs else float("nan"))
        gt_to_pred.extend(_nearest_distances(gb, pb))
        pred_to_gt.extend(_nearest_distances(pb, gb))
        _, matched = segment_f1_at_iou(psegs, gsegs, 0.5)
        coverage.append(matched / len(gsegs) if gsegs else 0.0)

    def _mean(xs: list[float]) -> float | None:
        xs = [x for x in xs if not math.isnan(x)]
        return round(statistics.mean(xs), 4) if xs else None

    return {
        "n_videos": len(videos),
        "n_pred_segments": n_pred_total,
        "n_gt_segments": n_gt_total,
        "boundary_f1": {f"@{t}s": _mean([f1 for _, _, f1 in b_prf[t]]) for t in tolerances},
        "boundary_precision": {f"@{t}s": _mean([p for p, _, _ in b_prf[t]]) for t in tolerances},
        "boundary_recall": {f"@{t}s": _mean([r for _, r, _ in b_prf[t]]) for t in tolerances},
        "segment_f1_at_iou": {f"@{thr}": _mean(s_f1[thr]) for thr in iou_thresholds},
        "over_segmentation_ratio": _mean(over_seg),
        "boundary_dist_gt_to_pred_s": _mean(gt_to_pred),
        "boundary_dist_pred_to_gt_s": _mean(pred_to_gt),
        "coverage_at_iou0.5": _mean(coverage),
    }


# ── Pretty printing ─────────────────────────────────────────────────────────────


def _fmt(v: Any) -> str:  # noqa: ANN401
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_report(report: dict[str, Any], tolerances: list[float], iou_thresholds: list[float]) -> None:
    """Print a per-detector comparison table (ranked by boundary F1 @ tightest tolerance)."""
    tol_key = f"@{tolerances[0]}s"
    iou_key = f"@{iou_thresholds[-1]}"
    rows = sorted(
        report.items(),
        key=lambda kv: (kv[1].get("boundary_f1", {}).get(tol_key) or -1),
        reverse=True,
    )
    cols = (
        f"{'detector':>16} | {'nVid':>5} | {'bF1' + tol_key:>9} | {'bP' + tol_key:>9} | "
        f"{'bR' + tol_key:>9} | {'segF1' + iou_key:>10} | {'overSeg':>8} | {'distGT→P':>9} | {'cov@.5':>7}"
    )
    print("\n" + "=" * len(cols))
    print("SHOT-BOUNDARY EVALUATION — detectors ranked by boundary F1 (higher = better)")
    print("=" * len(cols))
    print(cols)
    print("-" * len(cols))
    for name, d in rows:
        if not d.get("n_videos"):
            print(f"{name[:16]:>16} | no shared videos with GT")
            continue
        print(
            f"{name[:16]:>16} | {d['n_videos']:>5} | {_fmt(d['boundary_f1'][tol_key]):>9} | "
            f"{_fmt(d['boundary_precision'][tol_key]):>9} | {_fmt(d['boundary_recall'][tol_key]):>9} | "
            f"{_fmt(d['segment_f1_at_iou'][iou_key]):>10} | {_fmt(d['over_segmentation_ratio']):>8} | "
            f"{_fmt(d['boundary_dist_gt_to_pred_s']):>9} | {_fmt(d['coverage_at_iou0.5']):>7}"
        )
    print("=" * len(cols))


# ── CLI ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse args, evaluate each detector run against GT, print + write the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR", help="Detector output dir(s).")
    ap.add_argument("--gt-source", choices=["agibot"], default="agibot", help="GT segment source.")
    ap.add_argument("--gt-task-info", type=Path, required=True, help="AgiBot task_info dir (task_*.json).")
    ap.add_argument("--tolerances", type=float, nargs="+", default=[0.5, 1.0], help="Boundary match tolerances (s).")
    ap.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.5], help="Segment IoU thresholds.")
    ap.add_argument("--out", type=Path, default=None, help="Write the full JSON report here.")
    args = ap.parse_args()

    report: dict[str, Any] = {}
    for spec in args.run:
        if "=" not in spec:
            ap.error(f"--run must be NAME=DIR, got {spec!r}")
        name, _, path = spec.partition("=")
        run_dir = Path(path)
        pred_segs, fps = load_predicted_segments(run_dir)
        gt_segs = load_gt_segments_agibot(args.gt_task_info, fps)
        print(f"[loaded] {name}: {len(pred_segs)} videos (pred), {len(gt_segs)} videos (GT) from {run_dir}")
        report[name] = evaluate_run(pred_segs, gt_segs, args.tolerances, args.iou_thresholds)

    print_report(report, args.tolerances, args.iou_thresholds)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
