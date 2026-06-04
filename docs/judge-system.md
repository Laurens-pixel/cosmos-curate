# In-Pipeline Caption Judge System

After captioning, a `JudgeStage` scores each window's caption as **correct** or **incorrect** using a pluggable VLM judge. Verdicts are written alongside captions in `v0/all_window_judgments.json`.

---

## Quick start

```bash
pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path /config/input_videos \
  --output-clip-path /config/output_clips \
  --embedding-algorithm cradio \
  --evaluate \
  --judge-model vci_7b \
  --judge-gt-source none
```

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--evaluate` | off | Enable the JudgePhase |
| `--judge-model` | — | Judge variant (see table below). Comma-separate for multiple: `vci_7b,gemma4_e4b` |
| `--judge-gt-source` | `none` | Ground-truth source: `agibot`, `manual`, `none` |
| `--judge-gt-task-info-dir` | — | Required when `--judge-gt-source agibot` |
| `--judge-gt-annotation-path` | — | Required when `--judge-gt-source manual` |
| `--judge-caption-source` | captioning algorithm | Which caption key to judge |
| `--judge-prompt-variant` | `lenient_binary` | Prompt template |
| `--judge-batch-size` | 8 | Items per `judge_batch()` call — increase for throughput |
| `--judge-max-new-tokens` | 32 | Max tokens generated per verdict |

---

## Available judge variants

| Variant | Model | Evidence | VRAM | F1 (AgiBotWorld) |
|---|---|---|---|---|
| `vci_7b` | VCInspector-7B (LoRA Qwen2.5-VL) | MP4 bytes | 16 GB | **0.726** ← best |
| `vci_3b` | VCInspector-3B (LoRA Qwen2.5-VL) | MP4 bytes | 7.5 GB | 0.558 |
| `gemma4_e4b` | google/gemma-4-E4B-it | Text | 9 GB | 0.655 |
| `gemma4_e4b_video` | google/gemma-4-E4B-it | MP4 bytes | 9 GB | — |
| `gemma4_31b` | google/gemma-4-31B-it | Text | ~60 GB | — |

**Evidence types:**
- `text` — judge receives only the caption text and (optionally) a ground-truth action label
- `mp4_bytes` — judge receives the actual video clip frames decoded from the window MP4

**GPU budget note:** With VllmCaptionStage (1.0 GPU) + JudgeStage (0.5–1.0 GPU), total ≥ 2.0 GPU. Use `--gpus=2` for STREAMING mode. With `--gpus=3`, cosmos-xenna auto-scales vLLM to 2 workers (each loading the full KV cache) while JudgeStage stays at 1 worker — wasteful. Stick to 2 GPUs for caption + judge pipelines.

---

## Output format

`{output_dir}/v0/all_window_judgments.json` — one entry per window:

```json
{
  "clip_uuid": "abc123",
  "source_video": "task_327_ep001_head.mp4",
  "start_frame": 0,
  "end_frame": 255,
  "caption": "A robotic arm picks up a cucumber from the display.",
  "judge_variant": "vci_7b",
  "verdict": "incorrect",
  "score": 3,
  "explanation": "The arm appears to handle a bell pepper, not a cucumber.",
  "gt_action_text": "Pick bell pepper",
  "caption_source": "qwen"
}
```

`verdict` is `"correct"` when `score == 5`, `"incorrect"` otherwise. `gt_action_text` is `null` when `--judge-gt-source none`.

---

## Adding a new judge (3 files)

### 1. Create `cosmos_curate/models/judge_<name>.py`

```python
from cosmos_curate.models.judge_plugin import JudgePlugin, JudgeItem, JudgeResult, JudgeConfig
import attrs

@attrs.define
class MyJudge(JudgePlugin):

    @staticmethod
    def variant() -> str:
        return "my_judge"

    evidence_kind: str = "mp4_bytes"   # or "text" or "frames"

    @property
    def resources(self):
        from cosmos_curate.core.interfaces.stage_interface import CuratorStageResource
        return CuratorStageResource(gpus=1.0, cpus=4)

    def setup(self, config: JudgeConfig) -> None:
        # Load model weights here — called once per Ray worker
        self._model = ...
        self._processor = ...

    def judge_batch(self, items: list[JudgeItem]) -> list[JudgeResult]:
        # MUST be a true batch — one model.generate() call for the whole list.
        # A for-loop here defeats the purpose and will bottleneck the pipeline.
        ...
```

See the **Batched inference pattern** section below for the full implementation template.

### 2. Edit `cosmos_curate/models/judge_model_ids.py`

```python
JUDGE_MODEL_IDS = {
    ...
    "my_judge": "org/model-name-on-huggingface",
}
```

### 3. Edit `cosmos_curate/models/judge_interface.py`

```python
from cosmos_curate.models.judge_my_name import MyJudge

_JUDGE_PLUGINS: dict[str, type[JudgePlugin]] = {
    ...
    "my_judge": MyJudge,
}
```

That's it. `JudgeStage` reads the variant from the CLI and looks it up in the registry — it never needs modification.

---

## Batched inference pattern

`--judge-batch-size` only helps if `judge_batch()` does true batched GPU inference. A sequential loop inside `judge_batch()` means one `model.generate()` call per item regardless of batch size, which is the primary bottleneck in production runs.

### For `mp4_bytes` judges (video VLM)

```python
def _build_text(self, frames: list[Image.Image], caption: str) -> str:
    # IMPORTANT: pass actual frames here, not an empty list.
    # apply_chat_template uses the frames to insert the correct number
    # of video token markers. An empty list produces broken prompts.
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "text", "text": your_prompt_template.format(caption=caption)},
    ]}]
    return self._processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def judge_batch(self, items: list[JudgeItem]) -> list[JudgeResult]:
    results: list[JudgeResult | None] = [None] * len(items)
    valid_indices, valid_frames, valid_texts = [], [], []

    for i, item in enumerate(items):
        frames = decode_frames_from_mp4(item.mp4_bytes)  # your decode helper
        if not frames:
            results[i] = JudgeResult(verdict=None, score=5, explanation="no frames")
            continue
        valid_indices.append(i)
        valid_frames.append(frames)
        valid_texts.append(self._build_text(frames, item.caption))

    if not valid_indices:
        return [r or JudgeResult(verdict="correct", score=5, explanation="") for r in results]

    # Pad all clips to the same frame count (required by Gemma4; optional for Qwen)
    n_frames = max(len(f) for f in valid_frames)
    padded = [f + [f[-1]] * (n_frames - len(f)) for f in valid_frames]

    # Qwen2.5-VL based models need left-padding for batched generation
    self._processor.tokenizer.padding_side = "left"

    inputs = self._processor(
        text=valid_texts,
        videos=padded,
        padding=True,
        return_tensors="pt",
        num_frames=n_frames,   # Gemma4 only: overrides class-level default of 32
    ).to(self._device)

    with torch.no_grad():
        generated = self._model.generate(**inputs, max_new_tokens=32, do_sample=False)

    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)

    for orig_i, raw in zip(valid_indices, decoded):
        score, explanation = parse_score(raw)
        results[orig_i] = JudgeResult(
            verdict="correct" if score == 5 else "incorrect",
            score=score,
            explanation=explanation,
            raw_output=raw,
        )

    return [r or JudgeResult(verdict="correct", score=5, explanation="") for r in results]
```

### For `text` judges

Same idea but simpler — no frame decoding, just build `prompts = [template.format(...) for item in items]`, then one `processor(text=prompts, padding=True)` + one `model.generate()`.

---

## Performance notes

- **JudgeStage is the bottleneck** for short-clip datasets (YouCook2: 99.9% utilisation, AgiBotWorld: 80% at batch=16)
- Increasing `--judge-batch-size` from 4 → 16 reduced JudgeStage utilisation from 99.98% → 80% on AgiBotWorld (200 videos → 2,587 video run)
- Use `--judge-batch-size 16` as a default starting point; higher values may OOM on 40 GB GPUs with video judges
- For long-video datasets (>5 min each), VllmPrepStage becomes the bottleneck instead — judge batch size has no effect there
