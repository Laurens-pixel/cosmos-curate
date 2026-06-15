<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Downstream task performance

Three task families that measure whether curated clips + captions are *useful*, not just
well-formed.

## Retrieval (`retrieval.py`)
Given a text query (caption/instruction), find the matching clip — and vice versa.

```python
from downstream_eval.downstream.retrieval import evaluate_retrieval
# text_embeds, clip_embeds: (N, D) numpy arrays; row i of text matches row i of clips
out = evaluate_retrieval(text_embeds, clip_embeds, ks=(1, 5, 10))
# -> {"text_to_clip": {recall@1, recall@5, mrr, ndcg@5, ...}, "clip_to_text": {...}}
```

- Clip embeddings come straight from the pipeline (`load_clip_embeddings`, e.g. C-RADIO /
  cosmos-embed1 / InternVideo2).
- Text embeddings can come from the same multimodal encoder (cosmos-embed1 has a text tower)
  or any sentence encoder. Pass a `relevant_text_to_clip` list for many-to-many GT.

## Action recognition / classification (`action_recognition.py`)
Map a clip (embedding-derived scores or hard predictions) to an action/scene class.

```python
from downstream_eval.downstream.action_recognition import evaluate_action_recognition
out = evaluate_action_recognition(labels, scores=score_matrix, num_classes=C, ks=(1, 5))
# -> {top1_accuracy, top5_accuracy, mean_per_class_accuracy, macro_f1}
```

`mean_per_class_accuracy` and `macro_f1` are the imbalance-robust headline numbers
(InHARD/AgiBot class frequencies are skewed). A simple, training-free predictor is
nearest-class-centroid in embedding space (see `run_smoke.py`).

## Robot task completion (`robot_completion.py`)
Closed-loop policy evaluation. **This module scores rollout results; it does not train or
roll out a policy.**

```python
from downstream_eval.downstream.robot_completion import load_rollouts_from_json, robot_task_metrics
metrics = robot_task_metrics(load_rollouts_from_json("rollouts.json"))
# -> {task_success_rate, subtask_completion_rate, efficiency_ratio}
```

`rollouts.json` is a list of episodes:
```json
[{"success": true, "subtasks_completed": 4, "subtasks_total": 4, "steps": 110, "optimal_steps": 100}]
```

### Wiring LIBERO / LeRobot (external, not vendored)
The harness that produces `rollouts.json` lives outside this repo because it needs the sim +
policy stack. Recommended flow (no model training required if you reuse a released policy):

1. Build a `LeRobotDataset` from curated clips + captions joined with the dataset's actions
   (LIBERO/DROID/AgiBot ship proprioception + actions; the pipeline supplies the language).
2. Run / fine-tune a language-conditioned policy in the **LIBERO** simulator (deterministic,
   closed-loop) — or evaluate a released checkpoint as-is.
3. For each episode emit an `EpisodeResult` (success from the sim's goal check; subtasks from
   the task's predicate list; `optimal_steps` from the demo length).
4. Feed results here via `robot_task_metrics` / `RolloutProvider`.

`efficiency_ratio = optimal_steps / steps` (clipped to ≤1) is averaged over **successful**
episodes only, so it is not gamed by fast failures.
