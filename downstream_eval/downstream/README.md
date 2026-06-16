<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Downstream task performance

Each task is **actually performed** on the curated outputs and *then* evaluated. Every task
has two layers:

- a **metrics** module (pure-numpy scoring of results), and
- a **runner** module that builds the system, runs it, and calls the metrics.

| Task | Metrics (score only) | Runner (performs the task) |
| --- | --- | --- |
| Retrieval | `retrieval.py` | `retrieval_runner.py` |
| Action recognition | `action_recognition.py` | `action_recognition_runner.py` |
| Robot task completion | `robot_completion.py` | `policy_learning/` |

Text → vector encoding for the runners is provided by `encoders.py`
(`HashingEncoder` numpy fallback, optional `SentenceTransformerEncoder`, and
`CosmosEmbed1TextEncoder` which shares the pipeline's clip-embedding space).

## Retrieval (`retrieval_runner.py`)
Actually runs a nearest-neighbour search and reports R@K / MRR / nDCG@K.

```python
from downstream_eval.downstream.retrieval_runner import run_caption_retrieval, run_text_to_clip_retrieval
from downstream_eval.downstream.encoders import build_text_encoder, CosmosEmbed1TextEncoder

# text<->text: GT instruction (query) -> generated caption (gallery). Runs anywhere.
run = run_caption_retrieval(clips, references={clip_uuid: gt_instruction}, encoder=build_text_encoder("auto"))
print(run.metrics, run.top_k(0))

# cross-modal: text query -> clip video embedding (SAME space). Needs the shared-space encoder.
run = run_text_to_clip_retrieval(clip_ids, clip_embeddings, {"pick up the block": ["uuid1"]},
                                 encoder=CosmosEmbed1TextEncoder())
```

Pair the encoder with the embedding algorithm: use `CosmosEmbed1TextEncoder` only with
`cosmos-embed1` clip embeddings (shared space). For text↔text retrieval any encoder works.

## Action recognition / classification (`action_recognition_runner.py`)
Three ways to actually classify clips, each returning predictions + metrics
(`top1/top5`, `mean_per_class_accuracy`, `macro_f1`):

```python
from downstream_eval.downstream.action_recognition_runner import (
    run_nearest_centroid, run_linear_probe, run_zero_shot_language)

run_nearest_centroid(clip_embeddings, labels, class_names)   # train/test split, centroid match
run_linear_probe(clip_embeddings, labels, class_names)       # sklearn logistic probe
run_zero_shot_language(captions, labels, class_names)        # no training; caption<->class-name match
```

`mean_per_class_accuracy` / `macro_f1` are the imbalance-robust headline numbers
(InHARD/AgiBot class frequencies are skewed).

## Robot task completion (`policy_learning/`)
The full task: curated (clip+caption) + source-dataset actions → train a language-conditioned
policy → roll out in a closed-loop env → score. See `policy_learning/README.md`.

```python
# Runnable anywhere (no GPU/sim): synthetic env proves the rollout -> metrics path.
from downstream_eval.downstream.policy_learning.synthetic import run_synthetic_rollouts, ScriptedExpert
from downstream_eval.downstream.robot_completion import robot_task_metrics
print(robot_task_metrics(run_synthetic_rollouts(lambda rng: ScriptedExpert())))
```

`efficiency_ratio = optimal_steps / steps` (clipped to ≤1) is averaged over **successful**
episodes only, so it is not gamed by fast failures.
