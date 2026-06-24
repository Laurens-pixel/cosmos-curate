# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Text↔video caption-grounding benchmark (reference-free + ground-truth).

Complements ``benchmark_captions.py`` (which matches caption *text* to GT *text*). Here we
check whether a caption is supported by the **actual video frames**, using an image-text
model (SigLIP2 by default; CLIP also works). Two views are reported:

* **Reference-free** — ``caption_video_sim``: cosine between the caption embedding and the
  window's frame embedding. "Does the caption match what is on screen?" — needs no GT.
* **Ground-truth** — ``gt_video_sim``: cosine between the GT action text and the same
  frames. This is the *reference ceiling*: how well even the correct label grounds in this
  domain. ``caption_minus_gt`` says whether the caption aligns better or worse than the GT
  label itself.
* **Retrieval** — for each window, rank its own caption against all captions using the
  video as the query (and vice versa): median rank, R@1, R@10, and within-clip normalised
  rank. This measures discriminativeness, not just absolute similarity.

Honest caveat: web-trained image-text models are weak at fine-grained robot-manipulation
object discrimination (documented across InternVideo2 / ViCLIP / EMScore / LaViLa on this
data). Treat text↔video as a **secondary / diagnostic** signal — the VLM judge and the
text↔text semantic metrics remain primary.

Unlike ``benchmark_captions.py`` this needs torch + transformers + PyAV + a GPU and access
to the source videos, so it is a separate script you run as a job::

    python -m cosmos_curate.pipelines.video.evaluation.benchmark_textvideo \\
        --run qwen=/path/agibot_qwen_output \\
        --video-model /path/models/models--google--siglip2-so400m-patch14-384 \\
        --video-dir /path/agibot_head_full \\
        --out textvideo_report.json

Frame mapping: a window's source-video time span is
``duration_span[0] + frame / clip_framerate`` (read from ``metas/v0/<clip_uuid>.json``),
which correctly handles split clips — the same decoupling principle as the caption metric.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from cosmos_curate.pipelines.video.evaluation.benchmark_captions import WindowRec, load_run

_NUM_FRAMES = 8
_FRAME_SIZE = 384  # SigLIP2-SO400m patch14-384


# ── Clip metadata (source video + time span + framerate) ───────────────────────


def load_clip_meta(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Map clip_uuid → {source_video, duration_span, framerate} from metas/v0/*.json."""
    out: dict[str, dict[str, Any]] = {}
    meta_dir = run_dir / "metas" / "v0"
    if not meta_dir.is_dir():
        return out
    for p in meta_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        uuid = d.get("span_uuid") or p.stem
        out[str(uuid)] = {
            "source_video": d.get("source_video", ""),
            "duration_span": d.get("duration_span") or [0.0, 0.0],
            "framerate": d.get("framerate") or d.get("framerate_source") or 30.0,
        }
    return out


# ── Frame decoding (PyAV) ──────────────────────────────────────────────────────


def decode_window_frames(video_path: Path, start_s: float, end_s: float, num_frames: int) -> list[Any]:
    """Decode ``num_frames`` PIL frames evenly across [start_s, end_s] of the source video."""
    import av  # noqa: PLC0415 — heavy optional dep
    from PIL import Image  # noqa: PLC0415

    if end_s <= start_s:
        end_s = start_s + 0.1
    targets = [start_s + (end_s - start_s) * (i + 0.5) / num_frames for i in range(num_frames)]

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames: list[Any] = []
    ti = 0
    for frame in container.decode(stream):
        if ti >= len(targets):
            break
        t = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
        if t >= targets[ti]:
            img = frame.to_image().resize((_FRAME_SIZE, _FRAME_SIZE), Image.BILINEAR)
            frames.append(img)
            ti += 1
    container.close()
    # pad by repeating the last frame if the clip was short
    while frames and len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


# ── Vision-text encoder (SigLIP / CLIP) ────────────────────────────────────────


def build_vision_text_encoder(model_path: Path, model_kind: str) -> Any | None:  # noqa: ANN401
    """Load a SigLIP/CLIP image-text model; return an object with encode_images/encode_texts.

    Returns None (with a printed reason) if transformers/torch or the weights are missing, so
    the benchmark degrades gracefully. Resolves a HuggingFace-cache ``snapshots/<hash>/`` path
    automatically.
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415
    except ImportError as exc:
        print(f"[textvideo] torch/transformers unavailable ({exc}); cannot run")
        return None

    resolved = model_path
    snaps = model_path / "snapshots"
    if snaps.is_dir():
        cands = sorted(snaps.iterdir())
        if cands:
            resolved = cands[0]
    if not resolved.exists():
        print(f"[textvideo] model path {resolved} not found")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(str(resolved), local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(resolved), local_files_only=True)
    text_pad = "max_length" if model_kind == "siglip" else True

    class _Encoder:
        def encode_images(self, images: list[Any]) -> Any:  # noqa: ANN401
            inp = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype("float32")

        def encode_texts(self, texts: list[str]) -> Any:  # noqa: ANN401
            inp = processor(text=texts, padding=text_pad, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_text_features(**inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype("float32")

    print(f"[textvideo] loaded {model_kind} encoder from {resolved} on {device}")
    return _Encoder()


# ── Core scoring ───────────────────────────────────────────────────────────────


def _window_time_span(rec: WindowRec, meta: dict[str, Any]) -> tuple[float, float]:
    span = meta.get("duration_span") or [0.0, 0.0]
    fps = float(meta.get("framerate") or 30.0)
    return span[0] + rec.start_frame / fps, span[0] + rec.end_frame / fps


def score_run(
    recs: list[WindowRec],
    metas: dict[str, dict[str, Any]],
    video_dir: Path,
    encoder: Any,  # noqa: ANN401
    num_frames: int,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Embed each window's frames + caption + GT text, then compute similarity + retrieval."""
    import numpy as np  # noqa: PLC0415

    cap_embs: list[Any] = []
    gt_sims: list[float] = []
    clip_ids: list[str] = []
    n_skipped = 0

    cap_texts: list[str] = []
    gt_texts: list[str] = []
    keep: list[WindowRec] = []
    frame_batches: list[list[Any]] = []

    for r in recs:
        if not r.caption:
            continue
        meta = metas.get(r.clip_uuid)
        src = (meta or {}).get("source_video") or Path(r.source_video).name
        vpath = video_dir / Path(src).name
        if not meta or not vpath.exists():
            n_skipped += 1
            continue
        start_s, end_s = _window_time_span(r, meta)
        try:
            frames = decode_window_frames(vpath, start_s, end_s, num_frames)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[textvideo] decode failed {vpath}: {exc}")
            n_skipped += 1
            continue
        if not frames:
            n_skipped += 1
            continue
        frame_batches.append(frames)
        cap_texts.append(r.caption)
        gt_texts.append(r.gt_action_text or "")
        keep.append(r)

    if not keep:
        return {"n_scored": 0, "n_skipped": n_skipped}

    # Embed (batched per window for frames; one batch for all captions/gt).
    vid_embs = [encoder.encode_images(frames).mean(axis=0) for frames in frame_batches]
    vid = np.stack([v / (np.linalg.norm(v) + 1e-8) for v in vid_embs])
    cap = encoder.encode_texts(cap_texts)
    gt_mask = [bool(t.strip()) for t in gt_texts]
    gt = encoder.encode_texts([t if t.strip() else "." for t in gt_texts])

    for i in range(len(keep)):
        cap_embs.append(cap[i])
        clip_ids.append(keep[i].clip_uuid)
        if gt_mask[i]:
            gt_sims.append(float(gt[i] @ vid[i]))

    cap = np.stack([c / (np.linalg.norm(c) + 1e-8) for c in cap_embs])
    caption_video_sim = [float(cap[i] @ vid[i]) for i in range(len(keep))]

    # Retrieval: video query → all captions. sim matrix (N_videos x N_captions).
    sim = vid @ cap.T
    ranks: list[int] = []
    within_norm: list[float] = []
    r_at_1 = r_at_10 = 0
    clip_to_idx: dict[str, list[int]] = {}
    for i, cid in enumerate(clip_ids):
        clip_to_idx.setdefault(cid, []).append(i)
    for i in range(len(keep)):
        order = np.argsort(-sim[i])  # captions ranked for video i
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        r_at_1 += int(rank == 1)
        r_at_10 += int(rank <= 10)  # noqa: PLR2004
        # within-clip normalised rank
        peers = clip_to_idx[clip_ids[i]]
        if len(peers) > 1:
            peer_sims = [(j, float(sim[i, j])) for j in peers]
            peer_sims.sort(key=lambda x: -x[1])
            pos = [j for j, _ in peer_sims].index(i)
            within_norm.append(pos / (len(peers) - 1))

    n = len(keep)
    return {
        "n_scored": n,
        "n_skipped": n_skipped,
        "caption_video_sim_mean": round(statistics.mean(caption_video_sim), 4),
        "caption_video_sim_std": round(statistics.pstdev(caption_video_sim), 4) if n > 1 else None,
        "gt_video_sim_mean": round(statistics.mean(gt_sims), 4) if gt_sims else None,
        "caption_minus_gt": (
            round(statistics.mean(caption_video_sim) - statistics.mean(gt_sims), 4) if gt_sims else None
        ),
        "retrieval_median_rank": int(statistics.median(ranks)),
        "retrieval_r_at_1": round(r_at_1 / n, 4),
        "retrieval_r_at_10": round(r_at_10 / n, 4),
        "within_clip_norm_rank": round(statistics.mean(within_norm), 4) if within_norm else None,
        "pool_size": n,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Parse args, score each run's text↔video alignment, print + write the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR", help="Caption run(s). Repeatable.")
    ap.add_argument("--video-model", type=Path, required=True, help="SigLIP/CLIP model dir (or HF-cache dir).")
    ap.add_argument("--model-kind", choices=["siglip", "clip"], default="siglip", help="Text padding convention.")
    ap.add_argument("--video-dir", type=Path, required=True, help="Directory holding the source videos.")
    ap.add_argument("--num-frames", type=int, default=_NUM_FRAMES, help="Frames sampled per window (default 8).")
    ap.add_argument("--out", type=Path, default=None, help="Write the full JSON report here.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    encoder = build_vision_text_encoder(args.video_model, args.model_kind)
    if encoder is None:
        raise SystemExit(1)

    report: dict[str, Any] = {}
    for spec in args.run:
        if "=" not in spec:
            ap.error(f"--run must be NAME=DIR, got {spec!r}")
        name, _, path = spec.partition("=")
        run_dir = Path(path)
        recs = load_run(run_dir)
        metas = load_clip_meta(run_dir)
        print(f"[loaded] {name}: {len(recs)} windows, {len(metas)} clip metas from {run_dir}")
        report[name] = score_run(recs, metas, args.video_dir, encoder, args.num_frames, verbose=args.verbose)

    print("\n" + "=" * 110)
    print("TEXT↔VIDEO CAPTION GROUNDING (secondary signal — see caveat in module docstring)")
    print("=" * 110)
    hdr = (
        f"{'run':>14} | {'cap-vid':>8} | {'gt-vid':>8} | {'cap-gt':>8} | "
        f"{'medRank':>8} | {'R@1':>6} | {'R@10':>6} | {'wclip':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, d in report.items():
        if not d.get("n_scored"):
            print(f"{name[:14]:>14} | no windows scored (n_skipped={d.get('n_skipped')})")
            continue

        def _f(v: Any) -> str:  # noqa: ANN401
            return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

        print(
            f"{name[:14]:>14} | {_f(d['caption_video_sim_mean']):>8} | {_f(d['gt_video_sim_mean']):>8} | "
            f"{_f(d['caption_minus_gt']):>8} | {_f(d['retrieval_median_rank']):>8} | "
            f"{_f(d['retrieval_r_at_1']):>6} | {_f(d['retrieval_r_at_10']):>6} | {_f(d['within_clip_norm_rank']):>6}"
        )
    print("=" * 110)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
