# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Open-vocabulary object detection over caption windows.

Runs an open-vocabulary detector on the frames of each caption window and records which
objects are actually visible. The output, ``all_window_objects.json``, is the visual ground
truth that ``benchmark_captions.py`` uses for its object-grounding metric: it checks whether a
caption names the object that is *really on screen* (object recall) and whether the object it
names is actually there (object precision / swap detection) — the failure mode the semantic
metric smooths over, e.g. "cucumber" vs "bell pepper".

Detectors (``--detector``):
  * ``grounding_dino`` (default) — IDEA-Research Grounding DINO via HuggingFace transformers.
    The accurate choice: a text-conditioned grounding detector (~52 AP zero-shot COCO) that is
    far better than a fast YOLO head at telling visually-similar objects apart, which is exactly
    what object-grounding needs. Native to the transformers stack already used elsewhere here.
  * ``yolo_world`` — Ultralytics YOLO-World. The fast choice (real-time, ~35 AP LVIS) for
    scoring the full 15k-window set when accuracy is less critical than throughput.

This is a separate GPU pre-pass (not a metric inside the benchmark) so ``benchmark_captions.py``
stays dependency-free and CPU-only: it just reads this JSON, as it reads captions/judgments.

The candidate class list is a FIXED, curated object vocabulary (object_vocab.txt), independent
of the captions — the standard practice in open-vocabulary detection eval (cf. COCO-80 / LVIS).
Deriving classes from the captions being evaluated would be circular and would bias the metric,
so it is not done. To evaluate a different domain, pass a different ``--vocab-file``.

Example::

    python -m cosmos_curate.pipelines.video.evaluation.detect_objects \\
        --run /path/agibot_cosmos_r1_output \\
        --video-dir /path/agibot_head_2587 \\
        --detector grounding_dino \\
        --model /path/models/IDEA-Research/grounding-dino-base \\
        --out /path/agibot_cosmos_r1_output/v0/all_window_objects.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from cosmos_curate.pipelines.video.evaluation.benchmark_captions import (
    WindowRec,
    load_object_vocab,
    load_run,
)
from cosmos_curate.pipelines.video.evaluation.benchmark_textvideo import (
    decode_window_frames,
    load_clip_meta,
)

# The detector's class list is a FIXED, curated object vocabulary — like COCO-80 / LVIS in
# open-vocabulary detection eval — shipped as object_vocab.txt next to this module. It is
# deliberately NOT derived from the captions being evaluated: caption-derived classes would be
# circular (the detector could only find what the caption already named) and would bias
# recall/CHAIR. Swap the file (or pass --vocab-file) to evaluate a different domain.


# Fixed seed for the pre-shard shuffle. Keeping it constant means a time-budgeted (sampled)
# object pass draws the SAME random window sample for every detector, so comparisons between
# segmenters are never confounded by them being scored on different windows.
_SAMPLE_SEED = 1234


def load_vocab(vocab_file: Path | None) -> list[str]:
    """Load the detector's class prompts (canonical names + synonyms) from the curated vocab file.

    Uses the shared ``object_vocab.txt`` parser so the detector prompts with every surface form
    (e.g. "cart", "trolley", "shopping cart"); the benchmark later folds those onto one canonical
    when matching. Raises if the file yields no classes (an empty vocab → zero detections).
    """
    surface, _ = load_object_vocab(vocab_file)
    if not surface:
        msg = f"vocabulary file {vocab_file or 'object_vocab.txt'} contains no classes"
        raise ValueError(msg)
    return surface


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    """Split a vocabulary into prompt-sized chunks (Grounding DINO has a ~256-token limit)."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def build_grounding_dino_detector(
    model_path: Path,
    vocab: list[str],
    box_threshold: float,
    text_threshold: float,
    max_phrases_per_prompt: int,
    image_size: int = 480,
) -> Any | None:  # noqa: ANN401
    """Load Grounding DINO via HF transformers; None if unavailable.

    The vocabulary is chunked into several text prompts (token-limit safe) and every chunk is
    run per image; detections are merged by max score per matched phrase. Degrades gracefully
    (prints a reason, returns None) when transformers/torch or the weights are missing.
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: PLC0415
    except ImportError as exc:
        print(f"[detect] transformers/torch unavailable ({exc}); cannot run Grounding DINO")
        return None

    resolved = model_path
    snaps = model_path / "snapshots"
    if snaps.is_dir():
        cands = sorted(snaps.iterdir())
        if cands:
            resolved = cands[0]
    if not resolved.exists():
        print(f"[detect] model path {resolved} not found")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(resolved), local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(resolved), local_files_only=True)
    # Run the detector at a resolution matched to the signal the frames actually carry.
    #
    # GDINO defaults to 800x1333, and its cost scales with pixel count. Frames were previously
    # squashed to 384x384 upstream (a SigLIP2 constant) and then *upscaled* here to 800x1333 — so
    # ~2.6x the compute was spent on interpolated pixels carrying no new information, while the
    # squaring also destroyed the aspect ratio the detector relies on. Frames now arrive native, and
    # we resize once, here. Smaller AND more faithful than what it was doing before.
    processor.image_processor.size = {"shortest_edge": image_size, "longest_edge": int(image_size * 5 / 3)}
    # Grounding DINO expects lowercase phrases separated by " . " and a trailing period.
    prompts = [". ".join(chunk) + " ." for chunk in _chunked(vocab, max_phrases_per_prompt)]
    vocab_set = set(vocab)

    def _canonicalise(label: str) -> str | None:
        """Map a (possibly mangled) Grounding DINO phrase back to a real vocabulary entry.

        GDINO sometimes returns merged/partial spans of the prompt ("right right",
        "cart cart", "store supermarket"). We keep a detection only if its phrase maps cleanly
        to a vocab class: an exact match, or a vocab phrase that appears whole inside the
        returned span. Anything that doesn't map to a known object is dropped, which is what
        removes the fragment garbage.
        """
        label = label.strip().lower()
        if label in vocab_set:
            return label
        toks = label.split()
        # a single-word vocab entry present as one of the returned tokens
        for v in vocab_set:
            vtoks = v.split()
            if len(vtoks) == 1 and v in toks:
                return v
        # a multi-word vocab phrase contained contiguously in the returned span
        for v in vocab_set:
            if " " in v and v in label:
                return v
        return None
    print(
        f"[detect] loaded Grounding DINO from {resolved} on {device}; "
        f"vocabulary {len(vocab)} → {len(prompts)} prompt(s)/image"
    )

    def _post_process(outputs: Any, input_ids: Any, sizes: list[tuple[int, int]]) -> list[dict[str, Any]]:  # noqa: ANN401
        # The kwarg was renamed `box_threshold` → `threshold` across transformers versions.
        try:
            return processor.post_process_grounded_object_detection(
                outputs, input_ids, threshold=box_threshold, text_threshold=text_threshold, target_sizes=sizes
            )
        except TypeError:
            return processor.post_process_grounded_object_detection(
                outputs, input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=sizes
            )

    class _Detector:
        def detect(self, images: list[Any]) -> list[dict[str, float]]:
            """Return, per image, a {matched_phrase: max_confidence} dict.

            Every image is run once per vocabulary chunk, so cost is
            ``len(images) x len(prompts)`` forward passes. Both factors are minimised elsewhere
            (larger cross-window image batches; fewer, fuller prompts), and the matmuls run under
            autocast — this stage was previously the single slowest thing in the pipeline, taking
            longer than segmenting + captioning + judging 100 videos combined.
            """
            if not images:
                return []
            per_image: list[dict[str, float]] = [{} for _ in images]
            sizes = [(im.height, im.width) for im in images]
            for prompt in prompts:
                inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    outputs = model(**inputs)
                results = _post_process(outputs, inputs["input_ids"], sizes)
                for i, res in enumerate(results):
                    # transformers renamed string labels: "text_labels" (new) vs "labels" (old).
                    labels = res.get("text_labels")
                    if labels is None:
                        labels = res.get("labels", [])
                    scores = res.get("scores")
                    score_list = scores.tolist() if hasattr(scores, "tolist") else list(scores or [])
                    for lab, sc in zip(labels, score_list, strict=False):
                        name = _canonicalise(str(lab))
                        if name and float(sc) > per_image[i].get(name, 0.0):
                            per_image[i][name] = float(sc)
            return per_image

    return _Detector()


def build_yolo_world_detector(model_path: Path, vocab: list[str], conf: float) -> Any | None:  # noqa: ANN401
    """Load Ultralytics YOLO-World (fast path); None if unavailable."""
    try:
        import torch  # noqa: PLC0415
        from ultralytics import YOLOWorld  # noqa: PLC0415
    except ImportError as exc:
        print(f"[detect] ultralytics/torch unavailable ({exc}); install with "
              "`pip install ultralytics --no-cache-dir --no-user`")
        return None
    if not model_path.exists():
        print(f"[detect] model weights {model_path} not found")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOWorld(str(model_path))
    model.set_classes(vocab)
    print(f"[detect] loaded YOLO-World from {model_path} on {device}; vocabulary size {len(vocab)}")

    class _Detector:
        def detect(self, images: list[Any]) -> list[dict[str, float]]:
            if not images:
                return []
            results = model.predict(images, conf=conf, device=device, verbose=False)
            per_image: list[dict[str, float]] = []
            for res in results:
                found: dict[str, float] = {}
                boxes = getattr(res, "boxes", None)
                if boxes is not None and boxes.cls is not None:
                    for ci, cf in zip(boxes.cls.tolist(), boxes.conf.tolist(), strict=True):
                        name = str(res.names[int(ci)]).lower()
                        if cf >= conf and cf > found.get(name, 0.0):
                            found[name] = float(cf)
                per_image.append(found)
            return per_image

    return _Detector()


def _record_window(out: dict[str, Any], rec: WindowRec, per_frame: list[dict[str, float]], n_frames: int) -> None:
    """Merge a window's per-frame detections (max confidence per class) into the output dict."""
    merged: dict[str, float] = {}
    for found in per_frame:
        for name, cf in found.items():
            if cf > merged.get(name, 0.0):
                merged[name] = cf
    out.setdefault(rec.source_video, {}).setdefault(rec.clip_uuid, {})[rec.window_key] = {
        "objects": sorted(merged, key=lambda k: merged[k], reverse=True),
        "scores": {k: round(v, 4) for k, v in merged.items()},
        "n_frames": n_frames,
    }


def detect_run(
    recs: list[WindowRec],
    clip_meta: dict[str, dict[str, Any]],
    detector: Any,  # noqa: ANN401
    video_dir: Path,
    num_frames: int,
    limit: int | None,
    batch_windows: int = 4,
    image_size: int = 480,
    time_budget_s: float | None = None,
) -> dict[str, Any]:
    """Detect objects per window and assemble the nested output dict.

    For each window, frames are decoded from the source video over the window's time span
    (``duration_span[0] + frame/framerate``), every frame is run through the detector, and the
    per-frame detections are merged by taking the max confidence per class across frames.

    Frames from ``batch_windows`` windows are detected in **one** batched call rather than one call
    per window: a single window is only ``num_frames`` images, which leaves an A100 mostly idle, and
    this stage was the pipeline's dominant cost. Detections are sliced back out per window
    afterwards, so batching changes throughput only — never the result.

    ``time_budget_s`` stops the sweep cleanly once the budget is spent. This is safe *because*
    ``recs`` is shuffled before sharding (see ``main``): the windows completed within any budget are
    therefore a **uniform random sample** of all windows, so ``object_precision`` / ``CHAIR`` /
    ``object_recall`` — which are all means over windows — remain **unbiased estimates**, just with
    a wider confidence interval. A bigger budget buys precision, never correctness. This is what
    lets the detector be the *last* thing in the pipeline and still never overrun the job: it
    degrades to a smaller sample rather than dying and taking the caption report with it.
    """
    import torch  # noqa: PLC0415 — heavy optional dep, imported on use (as elsewhere in this file)

    started = time.monotonic()
    out: dict[str, Any] = {}
    n_done = 0
    pending: list[tuple[WindowRec, list[Any]]] = []

    def _detect_group(group: list[tuple[WindowRec, list[Any]]]) -> None:
        """Detect one group in a single batched call, halving the batch on OOM."""
        nonlocal n_done
        if not group:
            return
        try:
            per_image = detector.detect([f for _, frames in group for f in frames])
        except torch.OutOfMemoryError:
            # Never let one oversized batch kill the shard: split and retry. Frames are already
            # bounded, so this is a safety net rather than the primary defence.
            torch.cuda.empty_cache()
            if len(group) == 1:
                print(f"[detect] OOM on a single window; skipping {group[0][0].clip_uuid}", flush=True)
                return
            mid = len(group) // 2
            _detect_group(group[:mid])
            _detect_group(group[mid:])
            return
        cursor = 0
        for rec, frames in group:
            _record_window(out, rec, per_image[cursor : cursor + len(frames)], len(frames))
            cursor += len(frames)
            n_done += 1

    def flush() -> None:
        if not pending:
            return
        _detect_group(list(pending))
        pending.clear()
        print(f"[detect] {n_done} windows processed", flush=True)

    budget_hit = False
    for r in recs:
        if limit is not None and n_done + len(pending) >= limit:
            break
        if time_budget_s is not None and (time.monotonic() - started) >= time_budget_s:
            budget_hit = True
            break
        meta = clip_meta.get(r.clip_uuid, {})
        src = meta.get("source_video") or Path(r.source_video).name
        video_path = video_dir / Path(src).name
        if not video_path.exists():
            continue

        span = meta.get("duration_span") or [0.0, 0.0]
        fps = float(meta.get("framerate") or 30.0)
        start_s = float(span[0]) + r.start_frame / fps
        end_s = float(span[0]) + r.end_frame / fps

        # Bound the frame here, preserving aspect ratio, to the detector's own working resolution.
        # Not the 384x384 square (distorts geometry, then gets upscaled to 800x1333 for nothing) and
        # not native (WGO mixes 320x180 with 2560x1440; a detector batch pads to the largest image
        # in it, so one 3.7 MP frame inflates all 32 and OOMs the GPU — which is exactly what
        # happened: the shards thrashed and ran ~47x slower than on the uniform-resolution smoke).
        frames = decode_window_frames(
            video_path, start_s, end_s, num_frames, bounded=(image_size, int(image_size * 5 / 3))
        )
        if not frames:
            continue
        pending.append((r, frames))
        if len(pending) >= max(1, batch_windows):
            flush()
    flush()
    elapsed = time.monotonic() - started
    if budget_hit:
        print(
            f"[detect] time budget ({time_budget_s:.0f}s) reached after {n_done}/{len(recs)} windows "
            f"({100 * n_done / max(len(recs), 1):.0f}%). Windows were shuffled before sharding, so this "
            f"is a uniform random sample — the object metrics stay unbiased, with a wider CI.",
            flush=True,
        )
    print(f"[detect] done: {n_done} windows with detections in {elapsed / 60:.1f} min")
    return out


def main() -> None:
    """Parse args, run the detector over a caption run, write all_window_objects.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="Caption run output dir (reads its windows + metas).")
    ap.add_argument("--video-dir", type=Path, required=True, help="Directory holding the source videos.")
    ap.add_argument(
        "--detector",
        choices=["grounding_dino", "yolo_world"],
        default="grounding_dino",
        help="Open-vocab detector (default grounding_dino — the accurate choice).",
    )
    ap.add_argument("--model", type=Path, required=True, help="Detector weights/dir (Grounding DINO dir or YOLO .pt).")
    ap.add_argument(
        "--vocab-file",
        type=Path,
        default=None,
        help="Curated object class list, one per line (default: the checked-in object_vocab.txt). "
        "Swap this to evaluate a different domain — do not derive classes from the captions.",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help=(
            "Frames sampled per window (default 4). Detector cost is linear in this. Objects persist "
            "across a ~4 s clip and detections are merged by MAX confidence across frames, so 4 "
            "well-spread frames capture object presence about as well as 8 at half the cost."
        ),
    )
    ap.add_argument("--box-threshold", type=float, default=0.25, help="Grounding DINO box threshold (default 0.25).")
    ap.add_argument("--text-threshold", type=float, default=0.20, help="Grounding DINO text threshold (default 0.20).")
    ap.add_argument(
        "--max-phrases-per-prompt",
        type=int,
        default=60,
        help=(
            "Grounding DINO: vocabulary phrases per text prompt. Every image is run once PER prompt, "
            "so this is a direct multiplier on cost: 30 gave 7 passes/image over a 201-class vocab, 60 "
            "gives 4. Grounding DINO's text encoder takes 256 tokens and these classes are 1-2 words, "
            "so 60 fits comfortably (default 60)."
        ),
    )
    ap.add_argument(
        "--time-budget-s",
        type=float,
        default=None,
        dest="time_budget_s",
        help=(
            "Wall-clock budget for this shard's sweep. On expiry it stops cleanly and writes what it "
            "has. Windows are shuffled before sharding, so a partial sweep is a UNIFORM RANDOM SAMPLE "
            "-- object_precision / CHAIR / object_recall stay unbiased, just with a wider CI. This is "
            "what lets the detector run last without ever overrunning the job."
        ),
    )
    ap.add_argument(
        "--detector-image-size",
        type=int,
        default=480,
        dest="detector_image_size",
        help=(
            "Shortest-edge resolution the detector runs at (default 480). Detector cost scales with "
            "pixel count; GDINO's 800 default was being spent upscaling already-downsampled frames."
        ),
    )
    ap.add_argument(
        "--batch-windows",
        type=int,
        default=4,
        help="Windows whose frames are detected in one batched call (default 4 => 4*num_frames images).",
    )
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO-World confidence threshold (default 0.05).")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of windows (debugging).")
    ap.add_argument("--out", type=Path, default=None, help="Output JSON (default <run>/v0/all_window_objects.json).")
    ap.add_argument(
        "--shard-index", type=int, default=0,
        help="This process's shard (0-based). Run N processes with --shard-count N, each on its "
        "own GPU (CUDA_VISIBLE_DEVICES), to parallelize the detector pass across a node's GPUs — "
        "this stage is a plain sequential script, not a Ray pipeline, so it does not auto-scale.",
    )
    ap.add_argument("--shard-count", type=int, default=1, help="Total shards (default 1 = no sharding).")
    args = ap.parse_args()

    recs = load_run(args.run)
    n_total = len(recs)
    # Shuffle with a FIXED seed before sharding. Two things depend on this:
    #   1. Every shard gets an unbiased mix of videos (sequential order clusters windows by video,
    #      so a truncated sequential sweep would over-represent whichever videos came first).
    #   2. It is what makes `--time-budget-s` statistically sound: the windows a shard completes
    #      within any budget are a uniform random sample, so the object metrics (all means over
    #      windows) stay unbiased however early we stop.
    # The seed is fixed, so the sample is reproducible across runs and across detectors — the
    # comparison between two segmenters is not confounded by them seeing different windows.
    random.Random(_SAMPLE_SEED).shuffle(recs)
    if args.shard_count > 1:
        recs = recs[args.shard_index :: args.shard_count]
    print(
        f"[detect] loaded {len(recs)}/{n_total} windows from {args.run}"
        + (f" (shard {args.shard_index}/{args.shard_count}, shuffled seed={_SAMPLE_SEED})" if args.shard_count > 1 else "")
    )

    vocab = load_vocab(args.vocab_file)
    print(f"[detect] vocabulary: {len(vocab)} object classes from "
          f"{args.vocab_file or 'built-in object_vocab.txt'}")

    if args.detector == "grounding_dino":
        detector = build_grounding_dino_detector(
            args.model,
            vocab,
            args.box_threshold,
            args.text_threshold,
            args.max_phrases_per_prompt,
            args.detector_image_size,
        )
    else:
        detector = build_yolo_world_detector(args.model, vocab, args.conf)
    if detector is None:
        raise SystemExit(1)

    clip_meta = load_clip_meta(args.run)
    out = detect_run(
        recs,
        clip_meta,
        detector,
        args.video_dir,
        args.num_frames,
        args.limit,
        args.batch_windows,
        args.detector_image_size,
        args.time_budget_s,
    )

    dest = args.out or (args.run / "v0" / "all_window_objects.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"[detect] wrote object detections → {dest}")


if __name__ == "__main__":
    main()
