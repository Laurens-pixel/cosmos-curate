# Subtask Boundary Segmentation — Implementation Spec

## Scope

Given one continuous episode (video + per-frame timestamps; shot/take boundaries are
already resolved upstream), produce **subtask time boundaries only**. No relabeling
pass, no prompt-tuning loop — use a fixed, hardcoded prompt.

Output:

```json
{"segments": [{"start_sec": 0.0, "end_sec": 4.5, "subtask": "grasp mug"}]}
```

`subtask` is a free side-effect label produced by the segmentation call itself —
do not build a separate labeling stage.

**Model:** reuse the exact VLM client/config already wired up for T-PIVOT. Do not
introduce a new model or provider.

## Pipeline

1. **Sample frames** every `0.5s` across the full episode.
2. **Burn in timestamp**: resize each sampled frame to a `224px` tile and render its
   timestamp (e.g. `12.5s`) as a pixel overlay directly on the tile, top-left corner.
   It must be visible in the image itself, not passed separately as a text/legend map.
3. **Pack into contact sheets**: 20 tiles per sheet, 5 columns x 4 rows, chronological
   order (~10s of video per sheet). One episode → as many sheets as needed.
4. **One VLM call per episode**: send *all* sheets for the episode together, in order,
   in a single request. Do not split into per-sheet/windowed calls — that invents
   extra boundaries at the split points.
5. **Prompt rules** (hardcode, no tuning needed):
   - Segment only completed manipulation/state-change events — object held, released,
     moved, container opened/closed, contents transferred — not every visible motion.
   - Do not split approach, grasp adjustment, repositioning, or retreat into their own
     segments unless the world state actually changes.
   - Do not merge distinct pick/place/open/close/pour/wipe events.
   - Most segments should be 2–10s; shorter is fine only for fast pick/place/open/close
     events.
   - Use the burned-in tile timestamps for `start_sec`/`end_sec`.
   - Return only the JSON object above, nothing else.
6. **Parse & validate**: valid JSON; `start_sec` non-decreasing; `end_sec > start_sec`;
   all timestamps within `[0, episode_duration]`. On failure, retry once with the
   specific error appended to the prompt. If it fails again, fail the episode loudly —
   don't silently drop or auto-repair it.

## Config

| Key | Default |
|---|---|
| `sample_interval_sec` | 0.5 |
| `tile_size_px` | 224 |
| `sheet_columns` / `sheet_rows` | 5 / 4 |
| `duration_prior_range_sec` | [2, 10] |
