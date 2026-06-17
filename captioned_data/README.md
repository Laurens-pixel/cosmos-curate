# Captioned Data

Caption and judge outputs from the cosmos-curate pipeline on two robot manipulation datasets.

## Structure

```
captioned_data/
├── agibot_alpha/
│   ├── captions.json      # 15,301 Qwen 2.5-VL-7B captions (2,587 videos · 36 tasks)
│   ├── judgments.json     # VCInspector-7B + Gemma4-E4B verdicts for every window
│   └── examples/          # One representative video for the 3 GT-annotated tasks
│       ├── task_327_pickup_items_in_the_supermarket.mp4
│       ├── task_352_open_the_fridge_to_get_food.mp4
│       └── task_354_pickup_items_in_shelf.mp4
└── libero_object/
    ├── captions.json      # 454 Qwen 2.5-VL-7B captions (one per episode)
    ├── judgments.json     # VCInspector-7B + Gemma4-video verdicts
    └── examples/          # One video per task (10 tasks)
        ├── task1_orange_juice.mp4
        ├── task2_ketchup.mp4
        └── ...
```

## AgiBotWorld-Alpha (`agibot_alpha/`)

**Pipeline**: C-RADIOv4-H embeddings + Qwen 2.5-VL-7B captions, 256-frame windows at 2 fps (~8.5 s each).
**Scale**: 2,587 head-camera videos across 36 manipulation tasks (supermarket, fridge, laundry, restaurant, e-commerce, warehouse, etc.).

### `captions.json`

Array of objects, one per caption window:

```json
{
  "clip_uuid": "...",
  "source_video": "327_685046_head_color.mp4",
  "start_frame": 0,
  "end_frame": 255,
  "caption": "The robotic arm reaches into a tray of vegetables...",
  "caption_source": "qwen"
}
```

### `judgments.json`

Array of objects, one per judge call per window (two entries per window: `vci_7b` and `gemma4_e4b`):

```json
{
  "clip_uuid": "...",
  "source_video": "327_685046_head_color.mp4",
  "start_frame": 0,
  "end_frame": 255,
  "caption": "...",
  "judge_variant": "vci_7b",
  "verdict": "incorrect",
  "score": 2,
  "explanation": "The caption identifies the object as a tomato, but the robot is retrieving a shiitake mushroom.",
  "gt_action_text": "Retrieve shiitake mushroom from the shelf.",
  "caption_source": "qwen"
}
```

**Judge summary** (15,301 windows):

| Judge | Correct | Incorrect |
|-------|---------|-----------|
| VCInspector-7B (video, ref-free) | 23.5% | 76.5% |
| Gemma4-E4B (text, with GT label) | 40.4% | 59.6% |
| Both incorrect (high-confidence) | — | 48.3% |

**Example videos**: The three tasks included have full expert annotation — every window is manually labelled correct/incorrect (see `eval_stage/annotation_ground_truth.json` in the repo root). Tasks 327, 352, and 354 were used for judge benchmarking across VCInspector, Gemma4, Qwen-judge, LLaVA-Critic, LLaVA-Video, and EMScore.

The remaining 33 AgiBot tasks are available on the Snellius HPC cluster at
`/gpfs/work4/0/prjs0951/Sem/agibot_head_2587/`.

---

## LIBERO-Object (`libero_object/`)

**Pipeline**: Fixed-stride splitting (whole episode = one clip, ~15 s), `robosuite` prompt variant,
VCInspector-7B + Gemma4-video judges with `lenient_binary_no_gt` prompt (no GT text passed to judges).
**Scale**: 454 episodes × 10 pick-and-place tasks (~45 episodes/task), Panda arm, 256×256 images.

### `captions.json`

One caption window per episode (the whole episode is one clip):

```json
{
  "clip_uuid": "...",
  "source_video": "episode_0_task20.mp4",
  "start_frame": 0,
  "end_frame": 149,
  "caption": "The robotic arm grasps a small orange box and places it into a woven basket.",
  "caption_source": "qwen"
}
```

### `judgments.json`

Two judge entries per episode (`vci_7b` and `gemma4_e4b_video`):

```json
{
  "clip_uuid": "...",
  "source_video": "episode_0_task20.mp4",
  "judge_variant": "vci_7b",
  "verdict": "incorrect",
  "score": 2,
  "explanation": "The caption says 'orange box' but the robot is picking up an orange juice bottle.",
  "caption_source": "qwen"
}
```

**Judge summary** (454 windows):

| Judge | Correct | Incorrect |
|-------|---------|-----------|
| VCInspector-7B | 82.4% | 17.6% (80 episodes) |
| Gemma4-video   | 99.8% | 0.2%  (1 episode)  |

The 80 VCI-flagged episodes were used to build the `qwen_vci_filtered` training condition
in the robot policy experiment (see `scripts/inject_libero_captions.py`).

### Tasks

| File | GT instruction |
|------|----------------|
| `ep0807_task20.mp4` | pick up the orange juice and place it in the basket |
| `ep0808_task21.mp4` | pick up the ketchup and place it in the basket |
| `ep0810_task22.mp4` | pick up the cream cheese and place it in the basket |
| `ep0811_task23.mp4` | pick up the bbq sauce and place it in the basket |
| `ep0813_task24.mp4` | pick up the alphabet soup and place it in the basket |
| `ep0814_task25.mp4` | pick up the milk and place it in the basket |
| `ep0816_task26.mp4` | pick up the salad dressing and place it in the basket |
| `ep0819_task27.mp4` | pick up the butter and place it in the basket |
| `ep0821_task28.mp4` | pick up the tomato sauce and place it in the basket |
| `ep0823_task29.mp4` | pick up the chocolate pudding and place it in the basket |

Filenames match the keys in `captions.json` for direct lookup.
