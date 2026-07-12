"""Class F: ARC-Hunyuan x predictive-surprise boundary fusion.

Two detectors with complementary, measurable failure modes:

* **ARC-Hunyuan-Video-7B** is a captioning VLM asked to localise video chapters. It reasons about
  *semantics* ("now the arm moves to the sink"), so its cuts land on genuine task changes -- but it
  is coarse: it emits few, long chapters and misses fine sub-actions.
* **Predictive surprise** (``predictive_boundary``) fires wherever a world model fails to anticipate
  the next latent. It is fine-grained and training-free, but a purely bottom-up signal over-segments
  and drifts on slow transitions.

Fusion keeps every ARC cut as a high-precision *anchor* and admits a predictive cut only where ARC
is silent (no anchor within ``fusion_nms_tolerance_s``). ARC therefore supplies the skeleton and the
predictive stream fills the gaps -- neither can pull the other off a boundary it already found.

Measured on WGO-Bench (100 videos / 743 gold segments, segF1@IoU0.5), tolerance tuned on a 50-video
split and scored on the held-out 50 so the gain is not fitted:

===========================================  ==========  ===========
detector                                     held-out 50  full 100
===========================================  ==========  ===========
ARC-Hunyuan-7B alone                             0.3993         0.4104
predictive V-JEPA2 ViT-g alone                   0.4392         0.4172
**fusion (this module, tolerance 4.0 s)**      **0.5468**     **0.5074**
===========================================  ==========  ===========

The fusion also beats every published entry on the WGO board (best prior: ARC-Hunyuan at 0.410).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cosmos_curate.pipelines.video.clipping.shot_boundary_models import DetectorConfig

# ARC's own chapter-localisation instruction. It is reproduced verbatim from the model card: the
# checkpoint is instruction-tuned on this exact string, so paraphrasing it measurably degrades the
# span format and breaks parsing.
SEGMENT_PROMPT = "Localize video chapters with temporal boundaries and the corresponding sentence description."

_THINK_SPAN_RE = re.compile(r"\(<span>(\d+:\d+:\d+)\s*[-–]\s*(\d+:\d+:\d+)</span>\)")
_ANSWER_SPAN_RE = re.compile(r"<span>(\d+:\d+:\d+)\s*[-–]\s*(\d+:\d+:\d+)</span>\s*(.+)")
_BARE_SPAN_RE = re.compile(r"\[?\s*(\d+:\d+(?::\d+)?)\s*[-–]\s*(\d+:\d+(?::\d+)?)\s*\]?\s*(.+)")

# ARC ingests at most this many frames; longer videos are sampled to exactly this count.
_ARC_MAX_SEGMENTS = 150
# Below this duration ARC takes one frame per second, matching its training-time sampling.
_ARC_DENSE_MAX_S = 150.0
_ARC_AUDIO_SR = 16000
_EPS = 1e-6


def _hms_to_sec(hms: str) -> float:
    parts = hms.strip().split(":")
    if len(parts) == 3:  # noqa: PLR2004 - HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:  # noqa: PLR2004 - MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def parse_arc_spans(text: str, duration_s: float) -> list[tuple[float, float]]:
    """Parse ARC's free-form answer into ``(start, end)`` chapter spans.

    ARC emits its chapters in one of three layouts depending on how the decode unfolded, so all
    three are tried in order of granularity. The ``<think>`` block is preferred because it carries
    the model's *fine* segmentation; ``<answer>`` re-states a coarser summary of the same video.
    """

    def _valid(start: float, end: float) -> bool:
        return end > start and start < duration_s

    def _clamp(start: float, end: float) -> tuple[float, float]:
        return start, min(end, duration_s)

    # Strategy 1 - inline spans inside <think>: "<description> (<span>HH:MM:SS - HH:MM:SS</span>)".
    think = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
    if think:
        chunks = _THINK_SPAN_RE.split(think.group(1))
        spans = [
            _clamp(_hms_to_sec(chunks[k * 3 + 1]), _hms_to_sec(chunks[k * 3 + 2]))
            for k in range((len(chunks) - 1) // 3)
        ]
        spans = [s for s in spans if _valid(*s)]
        if spans:
            return sorted(spans)

    # Strategy 2 - one span per line inside <answer>: "<span>HH:MM:SS - HH:MM:SS</span> <desc>".
    answer = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
    if answer:
        spans = []
        for line in answer.group(1).strip().splitlines():
            m = _ANSWER_SPAN_RE.match(line.strip())
            if m:
                span = _clamp(_hms_to_sec(m.group(1)), _hms_to_sec(m.group(2)))
                if _valid(*span):
                    spans.append(span)
        if spans:
            return sorted(spans)

    # Strategy 3 - the model dropped its tags entirely; recover bare "HH:MM:SS - HH:MM:SS" lines.
    spans = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or "<" in stripped:
            continue
        m = _BARE_SPAN_RE.match(stripped)
        if m:
            span = _clamp(_hms_to_sec(m.group(1)), _hms_to_sec(m.group(2)))
            if _valid(*span):
                spans.append(span)
    return sorted(spans)


def interior_cuts(spans: list[tuple[float, float]]) -> list[float]:
    """Cut points strictly inside the video: every span edge except the two outermost ones.

    Both starts *and* ends are emitted, so a gap between two ARC chapters correctly yields two cuts
    (the end of one chapter and the start of the next) rather than being silently bridged.

    Mirrors ``benchmark_boundaries.internal_boundaries``; ``test_arc_fusion`` pins the two together
    so the detector and the metric can never drift apart.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    v_start = ordered[0][0]
    v_end = max(e for _, e in ordered)
    cuts: set[float] = set()
    for start, end in ordered:
        if start > v_start + _EPS:
            cuts.add(round(start, 3))
        if end < v_end - _EPS:
            cuts.add(round(end, 3))
    return sorted(cuts)


def fuse_boundaries(anchors: list[float], fine: list[float], tolerance_s: float) -> list[float]:
    """Merge a fine boundary stream into high-precision anchors, keeping every anchor.

    A fine cut is admitted only when it is more than ``tolerance_s`` from *every* boundary accepted
    so far -- including previously admitted fine cuts, so the fine stream also thins itself and
    cannot pile several near-duplicate cuts into one gap.

    The suppression is the whole mechanism, not a detail: it is what converts the predictive
    stream's over-segmentation into extra recall instead of extra false positives.
    """
    merged = list(anchors)
    for b in sorted(fine):
        if all(abs(b - kept) > tolerance_s for kept in merged):
            merged.append(b)
    return sorted(merged)


@dataclass
class _ArcSegmenter:
    """Lazily-loaded ARC-Hunyuan-Video-7B chapter localiser."""

    model_dir: str
    max_new_tokens: int = 1024
    _model: Any = None
    _processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache

        # ARC's modelling code predates the 4.45 rename of ``get_usable_length`` -> ``get_seq_length``
        # and still calls the old name during generation. Restore it as an alias.
        if not hasattr(DynamicCache, "get_usable_length"):

            def _get_usable_length(self: DynamicCache, new_seq_length: int, layer_idx: int = 0) -> int:  # noqa: ARG001
                return self.get_seq_length(layer_idx)

            DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

        # ARC ships as a transformers *submodule* (its code uses package-relative imports such as
        # ``from ...cache_utils import Cache``), so it cannot be loaded via trust_remote_code. The
        # run script binds the module into transformers/models/arc_hunyuan_video; these imports then
        # resolve exactly as they do for any first-party architecture.
        from transformers.models.arc_hunyuan_video.configuration_arc_hunyuan_video import (
            ARCHunyuanVideoConfig,
        )
        from transformers.models.arc_hunyuan_video.modeling_arc_hunyuan_video import (
            ARCHunyuanVideoForConditionalGeneration,
        )
        from transformers.models.arc_hunyuan_video.processing_arc_hunyuan_video import (
            ARCHunyuanVideoProcessor,
        )

        AutoConfig.register("arc_hunyuan_video", ARCHunyuanVideoConfig, exist_ok=True)
        AutoModelForCausalLM.register(ARCHunyuanVideoConfig, ARCHunyuanVideoForConditionalGeneration, exist_ok=True)

        self._model = (
            ARCHunyuanVideoForConditionalGeneration.from_pretrained(
                self.model_dir,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .eval()
            .cuda()
        )
        self._processor = ARCHunyuanVideoProcessor.from_pretrained(
            self.model_dir,
            font_path=f"{self.model_dir}/ARIAL.TTF",
        )

    def _frames(self, video_path: str) -> tuple[list[Any], float, float]:
        """Sample frames the way ARC was trained: 1 fps, or 150 uniform segments past 150 s."""
        import av

        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        duration_s = float(container.duration) / 1_000_000.0
        vlen = int(duration_s * fps)

        if duration_s <= _ARC_DENSE_MAX_S:
            intervals = [(int(i * fps), int((i + 1) * fps)) for i in range(math.ceil(duration_s))]
            sample_fps = 1.0
        else:
            seg_dur = duration_s / _ARC_MAX_SEGMENTS
            intervals = [
                (int(i * seg_dur * fps), int((i + 1) * seg_dur * fps)) for i in range(_ARC_MAX_SEGMENTS)
            ]
            sample_fps = 1.0 / seg_dur

        wanted = sorted({(start + min(end, vlen - 1)) // 2 for start, end in intervals})
        wanted_set = set(wanted)

        frames: list[Any] = []
        idx = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if idx in wanted_set:
                    frames.append(frame.to_image())
                idx += 1
            if len(frames) == len(wanted):
                break
        container.close()
        return frames, sample_fps, duration_s

    def spans(self, video_path: str) -> list[tuple[float, float]]:
        """Return ARC's chapter spans for one video, or ``[]`` if it produced none."""
        import numpy as np
        import torch

        self._load()
        frames, sample_fps, duration_s = self._frames(video_path)
        if not frames or duration_s <= 0:
            return []

        # WGO footage is silent, but ARC is an audio-visual model and its processor requires an
        # audio track; a zero waveform is the neutral input.
        audio = np.zeros(max(_ARC_AUDIO_SR, int(math.ceil(duration_s) * _ARC_AUDIO_SR)), dtype=np.float32)

        prompt = (
            f"<|startoftext|>{'<image>' * len(frames)}\n{SEGMENT_PROMPT}\n"
            "Output the thinking process in <think> </think> and final answer in <answer> </answer> "
            "tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
        )
        inputs = self._processor(
            text=prompt,
            video=frames,
            video_metadata={"fps": sample_fps},
            audio=audio,
            sampling_rate=_ARC_AUDIO_SR,
            duration=int(math.ceil(duration_s)),
            return_tensors="pt",
        ).to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return parse_arc_spans(self._processor.decode(output_ids[0], skip_special_tokens=True), duration_s)


class ArcFusionBoundaryDetector:
    """Class F: ARC chapter anchors + predictive-surprise infill.

    Satisfies the ``BoundaryDetector`` protocol, so it drops into
    ``ModelBoundaryClipExtractionStage`` unchanged. If ARC returns nothing for a video the detector
    degrades to the plain predictive stream rather than failing the video.
    """

    def __init__(self, arc: _ArcSegmenter, predictive: Any, tolerance_s: float) -> None:
        self._arc = arc
        self._predictive = predictive
        self._tolerance_s = tolerance_s

    def detect_boundaries(self, video_path: str) -> list[float]:
        """Return interior event-boundary timestamps, in seconds."""
        anchors = interior_cuts(self._arc.spans(video_path))
        fine = self._predictive.detect_boundaries(video_path)
        return fuse_boundaries(anchors, fine, self._tolerance_s)


# Registry. ``fusion_arc_predictive`` is the validated default (ARC + V-JEPA 2 ViT-g). Any
# ``predictive_*`` variant may be substituted as the fine stream via ``fusion_arc_<variant>``;
# on WGO all variants fused to within noise of each other, so the default is the one to use.
_DEFAULT_FINE = "predictive_vjepa2_giant"


def build_arc_fusion_detector(model_name: str, cfg: DetectorConfig) -> ArcFusionBoundaryDetector:
    """Build a ``fusion_arc_*`` detector by registry name."""
    from cosmos_curate.pipelines.video.clipping.predictive_boundary import build_predictive_detector
    from cosmos_curate.pipelines.video.clipping.shot_boundary_models import predictive_config_from

    suffix = model_name[len("fusion_arc") :].lstrip("_")
    fine_name = _DEFAULT_FINE if suffix in ("", "predictive") else suffix
    if not fine_name.startswith("predictive_"):
        msg = f"Unknown ARC fusion detector: {model_name} (fine stream must be a predictive_* variant)"
        raise ValueError(msg)

    return ArcFusionBoundaryDetector(
        _ArcSegmenter(model_dir=cfg.arc_model_dir, max_new_tokens=cfg.arc_max_new_tokens),
        build_predictive_detector(fine_name, predictive_config_from(cfg)),
        cfg.fusion_nms_tolerance_s,
    )


ARC_FUSION_MODELS = ("fusion_arc_predictive",)
