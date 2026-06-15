# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Downstream evaluation suite for cosmos-curate clip + caption outputs.

This package evaluates the *usefulness* of the curation pipeline's outputs along
three axes that mirror the project taxonomy:

- ``segmentation`` — how well clip boundaries match ground-truth sub-task boundaries.
- ``captioning``   — how good the generated captions are (reference-based, model-based, temporal).
- ``downstream``   — task-level usefulness (retrieval, action recognition, robot task completion).

Design notes
------------
- Metric *math* is pure-numpy and dependency-light so it is unit-testable without GPUs.
- Model-based metrics (CLIPScore, BERTScore) and the LLM/policy adapters degrade
  gracefully when their heavy optional dependencies are not installed.
- Loaders parse the cosmos-curate output layout directly (``metas/v0/*.json``,
  ``v0/all_window_captions.json``, ``v0/all_window_judgments.json``, ``*_embd/*.pickle``)
  so no part of the main package needs to be imported to run an evaluation.
"""
