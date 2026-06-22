# Caption / Judge Evaluation & Benchmarking

This directory holds the in-pipeline caption-judging stage and the offline benchmark that
ranks caption and judge models.

## `benchmark_captions.py` — offline model ranking

Reads the JSON a run already writes (`v0/all_window_captions.json`,
`v0/all_window_judgments.json`, `processed_videos/*.json`) — **no GPU, no weights, no
network** — and produces a per-model scorecard + ranking. It is pure-stdlib so it runs in
any environment, including the plain `curator` conda env.

```bash
python -m cosmos_curate.pipelines.video.evaluation.benchmark_captions \
    --run qwen=/path/agibot_qwen_output \
    --run qwen3=/path/agibot_qwen3_output \
    --manual-gt /path/annotation_ground_truth.json \
    --fine-run /path/agibot_gt_windows_output \
    --out benchmark_report.json
```

Each `--run NAME=DIR` contributes one caption model; runs are ranked against each other.
Metrics (any metric lacking its inputs is reported `null` rather than crashing):

| Family | Metrics |
|---|---|
| Caption stats | count, mean/median/p95 words, unique-caption ratio, vocab size, top duplicate |
| Judges | per-judge flag rate; precision/recall/F1/accuracy vs `--manual-gt` (error-detection framing) |
| Cross-judge agreement | pairwise raw agreement + **Cohen's kappa** over windows judged by ≥2 judges |
| Multi-label grounding (unweighted) | `action_recall_unweighted`, `action_hit_rate`, `grounded_precision`, **`action_f1`**, **`boundary_recall`** — every action counts equally so a minor action is not drowned out |
| Multi-label grounding (other views) | `multilabel_coverage_recall` (temporal-weighted), `single_label_recall` + `transition_minority_recall` (legacy, for comparison), `uncovered_window_rate` |
| Sub-action completeness (no GT) | `subaction_completeness` — needs `--fine-run`; how fully a coarse caption covers finer/action-aligned sub-captions |
| Adjacent consistency (no GT) | object continuity, action continuity, contradiction rate, intra-clip consistency |
| Cross-model similarity | bag-of-words cosine between caption models on shared (video, window) |
| Compute efficiency | caption-stage compute s, s/caption, est. caption tokens/s, judge-stage compute s |

The metrics are deliberately reported **side by side** (unweighted set view, coverage-weighted
view, legacy single-label view, and the GT-free views) so you can compare which best reflects
caption quality on your data before settling on a headline number. The composite `quality`
column uses the unweighted `action_f1`.

### Why the grounding metrics changed (the temporal-grounding fix)

A 256-frame (~8.5 s) window routinely spans more than one annotated action (e.g. *pick*
then *place*). The legacy metric assigned the single **majority-overlap** GT label and
scored a caption correct only if it mentioned that one label. This:

* gave a caption that describes only the majority action **full credit** even when it
  misses the boundary action, and
* **penalised** a caption that correctly describes the transition.

The AgiBot GT source (`gt_sources/agibot.py`) now returns **every** overlapping action with
its per-action coverage in `gt_extras.all_actions`, plus `window_coverage`, `is_transition`
and `num_overlapping_actions`. Four complementary fixes use this:

1. **Unweighted set scoring so the minor action counts equally** (`action_f1`,
   `action_hit_rate`, `grounded_precision`, `boundary_recall`). The window's GT is the *set*
   of all its actions; recall counts each action equally (a 2-action window where the caption
   names only the majority scores 0.5, not "almost full credit"), precision penalises
   asserting actions that aren't there, and `boundary_recall` measures whether transition
   windows captured ≥2 actions. This is the most direct answer to "the minor action must
   count too". `multilabel_coverage_recall` (temporal-weighted) and `single_label_recall`
   (legacy) are kept beside it for comparison.

2. **The judge sees all actions.** `judge_stage.py` now passes the full action list (not just
   the majority) to text judges, and the robot prompt (`prompts.py: LENIENT_BINARY`) tells the
   judge the clip may contain several actions and not to penalise a caption for describing a
   minor / boundary action that genuinely occurred. Video judges (`mp4_bytes`) still watch the
   clip with GT withheld.

3. **`uncovered_window_rate`** — windows whose GT actions cover `< --coverage-threshold` of the
   window are *flagged, not scored against a forced label*, so novel/idle moments don't get
   silently mislabelled.

On older runs without `all_actions` every metric transparently falls back to the single GT action.

### GT-free signals (work on any dataset, no annotations)

* **Adjacent-window consistency** — whether consecutive window captions in a clip share
  entities/actions, and a contradiction rate. Sidesteps window/boundary misalignment entirely.
* **Sub-action completeness** (`--fine-run`) — point it at a finer-grained or action-aligned
  caption run of the same videos; for each coarse window it measures how fully the coarse
  caption reproduces the content of the fine sub-captions inside it. A low score means the
  coarse caption dropped a sub-action — the minor-action problem, detected with no GT.

### Action-aligned windows — the structural fix (recommended for generation)

The root cause is that fixed ~8.5 s windows straddle action boundaries. The pipeline already
supports captioning on **action-aligned windows** via `--gt-windows-source agibot
--gt-windows-task-info-dir <dir>`: each caption corresponds to exactly one GT action, so the
majority/minority problem disappears at the source. Recommended comparison workflow:

```bash
# coarse fixed-window run vs action-aligned run, with the action-aligned run also serving
# as the GT-free sub-action reference for the coarse run:
python -m cosmos_curate.pipelines.video.evaluation.benchmark_captions \
    --run fixed=/path/agibot_fixed_window_output \
    --run aligned=/path/agibot_gt_windows_output \
    --fine-run /path/agibot_gt_windows_output \
    --out compare.json
```

The action-aligned run should show higher `action_f1` / `boundary_recall` and the fixed-window
run a lower `subaction_completeness` — quantifying exactly how much the windowing hurts.

## Adding / tuning models — config-driven

Model and judge knobs (HF id, GPU/CPU request, frame sampling, prompt variant, batch size)
live in [`../../../models/configs/model_registry.yaml`](../../../models/configs/model_registry.yaml)
and are read via `cosmos_curate.models.model_registry`. Tune a judge without editing Python,
or override/extend any entry by pointing `COSMOS_MODEL_REGISTRY` at your own YAML.

## Judges

Registered in `cosmos_curate/models/judge_interface.py`. Current VLM judges:

| Variant | Model | Notes |
|---|---|---|
| `gemma4_e4b` / `_video` | google/gemma-4-E4B-it | text / video |
| `vci_3b`, `vci_7b` | dipta007/VCInspector-* | LoRA Qwen2.5-VL (ActivityNet) |
| `qwen3vl_30b`, `qwen3vl_30b_fp8` | Qwen/Qwen3-VL-30B-A3B-Instruct | **recommended** — MoE (~3B active), no fine-tune, emits a *structured* verdict (object + action correctness, not just 1–5) |

`qwen3vl_30b` is opt-in via `--judge-model qwen3vl_30b`; it needs the weights downloaded
first. It is a stronger, better-documented replacement for the VCInspector LoRA.
