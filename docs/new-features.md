# New Features

This fork adds four features on top of upstream cosmos-curate. Each is a CLI flag — no code changes needed to use them.

---

## 1. C-RADIOv4-H Embeddings (`--embedding-algorithm cradio`)

NVIDIA's [C-RADIO](https://github.com/NVlabs/RADIO) vision transformer (ViT-H backbone) as an alternative to InternVideo2.

| Property | C-RADIO | InternVideo2 (default) |
|---|---|---|
| Embedding dim | 2,560 | 1,024 |
| VRAM | ~2 GB | ~6 GB |
| Frames | 8 × 432×432 | 8 × 224×224 |
| Normalisation | L2 | L2 |
| Pixi env | `unified` | `default` |

**Usage:**
```bash
pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path /config/input_videos \
  --output-clip-path /config/output_clips \
  --embedding-algorithm cradio
```

**Required extra bind mount** (C-RADIO code is not in the original upstream container):
```
--bind ./cosmos_curate:/opt/cosmos-curate/cosmos_curate
```
This overlays the modified source directory into the read-only container.

**Output:** per-clip embeddings stored in `cradio_embd/` (`.pickle`, shape `1×2560`) and `cradio_embd_parquet/`.

**Key implementation detail:** The model must be loaded in `float32`, not `bfloat16`. The patch embedding layer has a dtype mismatch that causes silent wrong results in bfloat16. See `cosmos_curate/models/cradio.py`.

---

## 2. Gemma4 Direct Captioning (`--captioning-algorithm gemma4`)

Uses Google's `gemma-4-E4B-it` (4.5B active / 8B total MoE) directly via HuggingFace — no vLLM server required.

| Property | Gemma4 | Qwen (default) |
|---|---|---|
| VRAM | ~9 GB | ~31 GB |
| Requires vLLM | No | Yes |
| Frames per window | 16 | ~20 |
| Max new tokens | 512 | configurable |

**Usage:**
```bash
pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path /config/input_videos \
  --output-clip-path /config/output_clips \
  --embedding-algorithm cradio \
  --captioning-algorithm gemma4
```

**Prerequisites:**
1. Download the model: `huggingface-cli download google/gemma-4-E4B-it --local-dir ./models/google/gemma-4-E4B-it`
2. Set up `pip_overrides` — see [gemma4-setup.md](gemma4-setup.md)

**Model path inside container:** `/config/models/google/gemma-4-E4B-it/`

---

## 3. OpenAI and Gemini Captioning (API-based)

Sends each window's frames to a cloud API instead of running a local model. Useful when VRAM is limited or you want the highest caption quality.

### OpenAI GPT-4o

```bash
pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path /config/input_videos \
  --output-clip-path /config/output_clips \
  --captioning-algorithm openai \
  --openai-model-name gpt-4o
```

Sends 8 frames per window as `image_url` content blocks. Includes exponential backoff (15 s base, 60 s max, 6 retries) to handle rate limits.

### Gemini 2.5 Flash

```bash
pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path /config/input_videos \
  --output-clip-path /config/output_clips \
  --captioning-algorithm gemini \
  --gemini-model-name models/gemini-2.5-flash
```

Sends inline MP4 bytes for each window. Same retry logic as OpenAI.

**API key config** — add to `config.yaml`:
```yaml
openai:
  api_key: sk-...
google:
  api_key: AIza...
```

**Caption quality benchmark** (30 robot manipulation videos, 125 windows):

| Model | Accuracy | Cost |
|---|---|---|
| Qwen2.5-VL-7B (local) | 50% | GPU time only |
| GPT-4o | 73% | ~$0.10/video |
| Gemini 2.5 Flash | 76% | ~$0.05/video |

---

## 4. GT-Window Captioning (`--gt-windows-source`)

Instead of TransNetV2 shot detection + 256-frame sliding windows, use ground-truth action boundaries directly as caption windows. This isolates captioning quality from segmentation quality.

**Usage (AgiBotWorld dataset):**
```bash
pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path /config/input_videos \
  --output-clip-path /config/output_clips \
  --gt-windows-source agibot \
  --gt-windows-task-info-dir /config/agibot_task_info
```

The `task_info` directory must contain JSON files with `start_frame`/`end_frame` per action (AgiBotWorld format). Each JSON file corresponds to one task.

**How it works:** When `--gt-windows-source` is set, the pipeline forces `fixed-stride` splitting with a 3600-second clip length (making the whole video one clip), so ground-truth frame numbers align directly with clip frame numbers. TransNetV2 is bypassed entirely.

**Adding support for a new dataset:** Subclass `GTWindowProvider` in `cosmos_curate/pipelines/video/captioning/gt_window_provider.py` and register it in `_GT_PROVIDERS`.

---

## 5. Multi-View Captioning (`--multi-view`)

Groups multiple camera angles of the same scene into a single VLM inference call, producing one richer scene-level caption. Implemented for nuScenes-style datasets (3 front cameras per scene).

```bash
pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
  --input-video-path /config/nuscenes_videos \
  --output-clip-path /config/output_clips \
  --captioning-prompt-variant av-multiview \
  --multi-view
```

Videos must be named `scene-XXXX_CAM_FRONT.mp4`, `scene-XXXX_CAM_FRONT_LEFT.mp4`, `scene-XXXX_CAM_FRONT_RIGHT.mp4`. The stage groups them by scene ID and waits until all three cameras have been processed before submitting to the VLM.
