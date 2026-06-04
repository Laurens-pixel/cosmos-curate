<p align="center">
    <img src="docs/assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Product Website](https://www.nvidia.com/en-us/ai/cosmos/)

# Cosmos-Curate

---

## HPC Fork — What's new

This fork extends upstream cosmos-curate with four features and a full **Apptainer deployment guide** for Slurm HPC clusters.

| Feature | CLI flag | Doc |
|---|---|---|
| C-RADIOv4-H embeddings (ViT-H, 2560-d, ~2 GB VRAM) | `--embedding-algorithm cradio` | [docs/new-features.md](docs/new-features.md#1-cradiovh-embeddings) |
| Gemma4 / OpenAI GPT-4o / Gemini 2.5 Flash captioning | `--captioning-algorithm gemma4\|openai\|gemini` | [docs/new-features.md](docs/new-features.md#2-gemma4-direct-captioning) |
| In-pipeline VLM caption judge (pluggable, 5 variants) | `--evaluate --judge-model vci_7b` | [docs/judge-system.md](docs/judge-system.md) |
| GT-window captioning (bypass TransNetV2) | `--gt-windows-source agibot` | [docs/new-features.md](docs/new-features.md#4-gt-window-captioning) |

**Running on a Slurm cluster with Apptainer?** Start here: [docs/hpc-apptainer.md](docs/hpc-apptainer.md)

**Setting up Gemma4?** The container needs a `pip_overrides` directory: [docs/gemma4-setup.md](docs/gemma4-setup.md)

### Files added

| File | Description |
|---|---|
| `cosmos_curate/models/cradio.py` | Model wrapper for C-RADIOv4-H: loads weights, preprocesses frames, returns L2-normalised embeddings. |
| `cosmos_curate/pipelines/video/embedding/cradio_stages.py` | Two pipeline stages — `FrameCreationStage` (CPU) and `EmbeddingStage` (GPU ¼) — that run C-RADIO on each clip. |
| `cosmos_curate/pipelines/video/captioning/gemma4_direct_stage.py` | Single captioning stage that runs Gemma4-E4B-it directly via HuggingFace, replacing the two-stage vLLM approach. |
| `cosmos_curate/pipelines/video/captioning/gt_window_provider.py` | Abstract `GTWindowProvider` base class and `AgiBotGTWindowProvider` implementation that reads action boundaries from dataset JSON files. |
| `cosmos_curate/models/judge_plugin.py` | Abstract `JudgePlugin` base class defining the interface all judge variants must implement. |
| `cosmos_curate/models/judge_interface.py` | Registry dict and factory functions for looking up judge plugins by variant name. |
| `cosmos_curate/models/judge_model_ids.py` | Mapping from variant name to HuggingFace model ID for all judge variants. |
| `cosmos_curate/models/judge_gemma4.py` | Judge plugins `gemma4_e4b` and `gemma4_31b`: score captions using Gemma4 text-only inference. |
| `cosmos_curate/models/judge_gemma4_video.py` | Judge plugins `gemma4_e4b_video` and `gemma4_31b_video`: score captions by decoding and watching the actual video clip. |
| `cosmos_curate/models/judge_vci.py` | Judge plugins `vci_3b` and `vci_7b`: score captions using VCInspector (LoRA-tuned Qwen2.5-VL). |
| `cosmos_curate/models/judge_video_utils.py` | Shared utilities for decoding MP4 bytes into PIL frames used by all video-based judge plugins. |
| `cosmos_curate/pipelines/video/evaluation/judge_stage.py` | `JudgeStage`: judge-agnostic pipeline stage that looks up the requested plugin and calls `judge_batch()`. |
| `cosmos_curate/pipelines/video/evaluation/phases.py` | `JudgePhase` and `JudgePhaseConfig` wiring the judge stage into the split pipeline. |
| `cosmos_curate/pipelines/video/evaluation/gt_sources/` | Ground-truth source implementations (`agibot`, `inhard`, `manual`, `none`, `nuscenes`, `youcook2`) for optional judge grounding. |
| `cosmos_curate/models/vllm_gemma4.py` | vLLM plugin for Gemma4 used by the captioning stage. |
| `patches/monitoring_patched.py` | Patched `cosmos_xenna` monitoring module that fixes a `ConnectionError` when multiple Ray instances are active. |
| `cosmos-curate.def` | Apptainer container definition file for building the `.sif` image on HPC clusters. |
| `build_apptainer.sh` | Build script with quota monitoring to prevent accidental home-directory writes during the ~30 min build. |
| `run_split_annotate.sbatch` | Slurm batch job template for running the full pipeline on a GPU node. |
| `docs/hpc-apptainer.md` | Full deployment guide: Apptainer flags, bind mounts, Ray patch, memory requirements, STREAMING vs BATCH mode. |
| `docs/new-features.md` | Usage guide for all four new pipeline features with copy-pasteable example commands. |
| `docs/judge-system.md` | Judge plugin API reference, available variants, output format, and instructions for adding a new judge. |
| `docs/gemma4-setup.md` | Step-by-step recipe for creating the `pip_overrides/transformers/` directory needed by Gemma4. |

### Files modified

| File | Change |
|---|---|
| `cosmos_curate/pipelines/video/splitting_pipeline.py` | Added `--embedding-algorithm cradio`, `--captioning-algorithm gemma4/openai/gemini`, `--evaluate`, `--judge-model`, `--gt-windows-source`, and `--multi-view` CLI flags. |
| `cosmos_curate/pipelines/video/utils/data_model.py` | Added `cradio_frames`, `cradio_embedding`, `pipeline_start_ts`, and `stage_timestamps` fields to the `Clip` and `Video` data classes. |
| `cosmos_curate/pipelines/video/read_write/metadata_writer_stage.py` | Added C-RADIO embedding output and per-stage timestamp persistence to the metadata writer. |
| `cosmos_curate/pipelines/video/captioning/phases.py` | Wired in Gemma4, OpenAI, and Gemini captioning stages alongside the existing vLLM stage. |
| `cosmos_curate/pipelines/video/captioning/vllm_caption_stage.py` | Added multi-view scene buffering (`--multi-view`) and GT-window support (`--gt-windows-source`). |
| `cosmos_curate/pipelines/video/captioning/gemini_caption_stage.py` | Added exponential backoff and inline MP4 byte sending for the Gemini API. |
| `cosmos_curate/pipelines/video/captioning/openai_caption_stage.py` | Switched from `video_url` (vLLM extension) to `image_url` frame blocks for compatibility with the official OpenAI API. |
| `cosmos_curate/pipelines/video/clipping/clip_extraction_stages.py` | Carry `stage_timestamps` forward when creating per-chunk sub-tasks so timing data is not lost. |
| `cosmos_curate/pipelines/video/embedding/phases.py` | Registered the C-RADIO embedding phase alongside the existing InternVideo2 phase. |
| `cosmos_curate/models/prompts.py` | Added `av-multiview` prompt for multi-camera scene captioning. |
| `cosmos_curate/models/vllm_qwen.py` | Added `make_multiview_message()` and `LIMIT_MM_PER_PROMPT = {"video": 3}` for multi-view inference. |
| `cosmos_curate/models/vllm_plugin.py` | Added `make_multiview_llm_input()` abstract method to the vLLM plugin interface. |
| `cosmos_curate/configs/all_models.json` | Added `cradio_v4_h` model entry. |
| `cosmos_curate/core/utils/infra/performance_utils.py` | Minor instrumentation additions for per-stage timing. |

---

> **Everything below this line is from the original NVIDIA cosmos-curate repository:**
> [https://github.com/NVIDIA/cosmos-curator](https://github.com/NVIDIA/cosmos-curator)

---


A powerful video curation system that processes, analyzes, and organizes video content using advanced AI models and distributed computing.

## Important

Please run `git submodule sync` if you have cloned the repository before and just pulled the latest update.
We updated the URL for `cosmos-xenna` submodule on 08/04/2025.

## Overview

Cosmos-Curate is a comprehensive solution for video processing and curation using state-of-the-art AI models,
which powers the training data generation for [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) at NVIDIA.
It is built on top of a framework optimized for GPU-accelerated streaming pipeline,
which is now open-sourced independently as [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna).

## Features

- **Video Processing**: Efficient video splitting, annotation, filtering, deduplication, and dataset generation
- **AI-Powered Analysis**: Advanced video analysis using multiple model families
- **Distributed Computing**: Scalable processing using [Cosmos-Xenna](https://github.com/nvidia-cosmos/cosmos-xenna) built on top of [Ray](https://www.anyscale.com/product/open-source/ray)
- **Cloud Integration**: Support for various platforms
- **Pipeline System**: Modular and extensible pipeline architecture

## Documentation

Comprehensive documentation is available under [docs/](docs/README.md) directory.

### User Documentation
- [End User Guide](docs/client/END_USER_GUIDE.md) - instructions to setup environment and run data pipelines
- [Reference Video Pipelines Guide](docs/curator/REFERENCE_PIPELINES_VIDEO.md) - details for general video processing pipelines
- [Reference AV Pipelines Guide](docs/curator/REFERENCE_PIPELINES_AV.md) - details for multi-camera video, and (upcoming) GPS & LiDAR processing pipelines for autonomous vehicle (AV)
- [NVCF Guide](docs/client/NVCF_GUIDE.md) - deployment instruction on [Nvidia Cloud Functions](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/overview.html)

### Developer Documentation
- [Developer Guide](docs/DEVELOPER_GUIDE.md) - information for contributors
- [Architecture Guide](docs/curator/ARCHITECTURE_GUIDE.md) - diagrams and description to help understand core architecture
- [Pipeline Design Guide](docs/curator/PIPELINE_DESIGN_GUIDE.md) - detailed walk-through of the hello-world pipeline and performance optimization points
- [Observability Guide](docs/curator/OBSERVABILITY_GUIDE.md) - instructions to setup and understand monitoring dashboard

### AI Agent Context Files
- [AGENTS.md](AGENTS.md) - Context file for Codex
- [CLAUDE.md](CLAUDE.md) - Context file for Claude Code
- [GEMINI.md](GEMINI.md) - Context file for Gemini

## Directory Structure

```bash
cosmos-curate/
├── cosmos_curate/         # Curate implementation
│   ├── client              # CLI to run locally
│       ├── image_cli       # Docker image management
│       ├── local_cli       # Launch pipelines by running local container
│       ├── nvcf_cli        # Launch pipelines on NVIDIA cloud function
│       ├── slurm_cli       # Launch pipelines on Slurm cluster
│       ├── utils           # Common utilities for various CLI apps
│   ├── core/               # Core functionality
│       ├── cf              # Service entry point for a cloud function deployment
│       ├── interfaces      # Core base class to integrate model and define new pipelines
│       ├── managers        # CLIs to run inside the container to manage models, databases, etc.
│       ├── utils           # Common utilities for pipelines
│   ├── models/             # AI model inference implementations
│   ├── pipelines/          # Pipeline implementations
│       ├── examples/       # Minimal example pipelines to help understand the framework
│       ├── video/          # Reference pipelines for video curation
│   ├── scripts/            # Startup scripts in various deployment environments
├── cosmos-xenna            # Git submodule for https://github.com/nvidia-cosmos/cosmos-xenna
├── packages                # Dockerfiles and scripts related to packaging
│   ├── cosmos_curate       # Dockerfile template and conda environment recipes for building cosmos_curate image
├── tests                   # Tests for testing
│   ├── cosmos_curate             
│       ├── pipelines       # Tests for models and pipeline stages for cosmos_curate
│       ├── client          # Tests for client CLIs
├── examples                # Example configuration files and scripts
```

Note: To initialize and update the `cosmos-xenna` submodule, run:

```bash
git submodule update --init --recursive
```

This ensures all submodule content is checked out correctly.

## Support

For support and questions:
- Check the [documentation](docs/README.md)
- Open an issue on GitHub

## Acknowledgments

- [cosmos-xenna](https://github.com/nvidia-cosmos/cosmos-xenna) team for the core library
- All contributors and users of the project

## Responsible Use of AI Models
[Responsible Use](./RESPONSIBLE_USE.md)

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
