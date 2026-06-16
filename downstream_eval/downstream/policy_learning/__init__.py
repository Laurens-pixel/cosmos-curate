# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language-conditioned policy learning as a downstream task.

Pipeline: curated (clip + caption) joined with the source dataset's (actions, states)
-> LeRobotDataset -> train a language-conditioned diffusion policy -> roll out in a
closed-loop environment -> score with robot task-completion metrics.

- ``lerobot_adapter``: convert curated episodes into a ``LeRobotDataset`` (real, gated).
- ``train``          : train a diffusion policy on that dataset (real, gated on lerobot/torch).
- ``rollout``        : rollout harness protocol + a LIBERO implementation (real, gated).
- ``synthetic``      : pure-numpy reach env + policies so the rollout->metrics plumbing is
                       fully runnable and smoke-testable without a GPU/sim.
"""
