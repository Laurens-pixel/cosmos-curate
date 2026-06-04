# HPC Deployment Guide — Apptainer on Slurm

This guide covers running cosmos-curate on a Slurm HPC cluster using **Apptainer** (the HPC-native successor to Singularity). Tested on Snellius (SURF, Netherlands) with A100 40 GB GPUs. The same approach works on any cluster that has Apptainer ≥ 1.2 and Slurm.

---

## 1. Why Apptainer instead of Docker

HPC clusters do not allow Docker (it requires root). Apptainer runs as an unprivileged user, mounts GPU drivers via `--nv`, and packages the entire Python environment into a single read-only `.sif` squashfs image. This means:

- No internet access needed on compute nodes
- Reproducible — the image is immutable
- GPU passthrough with `--nv` (no CUDA installation needed on the host)
- All model weights and outputs live **outside** the container, mounted at runtime

---

## 2. Building the container

```bash
# Build from the definition file (takes ~30 min, produces ~20 GB .sif)
apptainer build cosmos-curate.sif cosmos-curate.def

# Lighter build without optional extras
apptainer build --build-arg SLIM=1 cosmos-curate.sif cosmos-curate.def
```

The `.def` file is at the root of this repo. It builds on top of the NVIDIA CUDA base image and installs the full pixi environment.

> **Storage tip**: Build on a scratch/work filesystem, not your home directory. The image is ~20 GB.

---

## 3. Workspace layout

The container is read-only. Everything mutable lives in a local workspace directory that gets bind-mounted into the container at `/config`:

```
cosmos_curate_local_workspace/
├── models/                  # Downloaded model weights (bind-mounted at /config/models/)
│   ├── Qwen/Qwen2.5-VL-7B-Instruct/
│   ├── nvidia/C-RADIOv4-H/
│   ├── OpenGVLab/InternVideo2-Stage2_1B-224p-f4/
│   ├── google/gemma-4-E4B-it/
│   ├── dipta007/VCInspector-7B/
│   └── Sn4kehead/TransNetV2/
├── output_clips/            # Pipeline outputs
├── tmp/                     # Isolated /tmp (see §5)
└── tmp_home/                # Writable home directory inside container
```

---

## 4. Critical Apptainer flags

Two flags are **required** and not obvious:

### `--no-mount hostfs`

Without this, Apptainer mounts the host's entire filesystem, which causes the host's `/opt` directory to overlay the container's `/opt/cosmos-curate/`. The pipeline fails silently because it can no longer find its own Python packages.

### `--no-mount tmp`

Without this, Apptainer mounts the host's `/tmp`. Ray writes its session files to `/tmp`, and multiple pipeline runs on the same node will conflict. Pair this with a custom `/tmp` bind mount (see §6).

---

## 5. Bind mount reference

| Host path | Container path | Purpose |
|---|---|---|
| `./cosmos_curate_local_workspace` | `/config` | Workspace root (models, outputs) |
| `./.config/cosmos_curate/config.yaml` | `/cosmos_curate/config/cosmos_curate.yaml` | HuggingFace token + API keys |
| `./cosmos_curate_local_workspace/tmp_home` | `/home/$USER` | Writable home (prevents Ray writing to read-only container home) |
| `./cosmos_curate_local_workspace/tmp` | `/tmp` | Isolated Ray session directory |
| `./input_videos` | `/config/input_videos` | Your input video files |
| `./patches/monitoring_patched.py` | `/opt/cosmos-curate/.pixi/envs/default/lib/python3.12/site-packages/cosmos_xenna/pipelines/private/monitoring.py` | Ray multi-instance fix (see §7) |
| `./cosmos_curate` | `/opt/cosmos-curate/cosmos_curate` | **Only for C-RADIO / new features**: overlays modified source into container |

The last mount is what makes the new features (C-RADIO, Gemma4 captioning, judge system) available inside the read-only container — the host's source directory shadows the container's copy.

---

## 6. Environment variables

| Variable | Value | Purpose |
|---|---|---|
| `TMPDIR` | `/tmp` | Tells Python to use the isolated tmp |
| `RAY_TMPDIR` | `/tmp` | Tells Ray to use the isolated tmp |
| `HF_HOME` | `/config/default_workspace/weights/hf_home/` | HuggingFace cache (inside container = workspace) |
| `COSMOS_CURATOR_LOCAL_DOCKER_JOB` | `1` | Tells cosmos-curate to run in single-node local mode |

---

## 7. Ray multi-instance monitoring patch

When cosmos-curate downloads models, it spawns a subprocess that starts its own Ray instance. The main pipeline then starts a second one. The monitoring code in `cosmos_xenna` calls `ray.util.state.list_actors()` without specifying which Ray instance to query, which raises:

```
ConnectionError: Found multiple active Ray instances
```

**Fix**: `patches/monitoring_patched.py` modifies `get_ray_actors()` to:
1. Get the current instance's GCS address via `ray.get_runtime_context().gcs_address`
2. Pass it explicitly: `list_actors(address=gcs_address)`
3. Catch `ConnectionError` as a fallback (returns empty list)

Bind-mount it on every pipeline run:
```
--bind ./patches/monitoring_patched.py:/opt/cosmos-curate/.pixi/envs/default/lib/python3.12/site-packages/cosmos_xenna/pipelines/private/monitoring.py
```

---

## 8. Full working example (single node, interactive)

Fill in `<YOUR_ACCOUNT>`, `<YOUR_PARTITION>`, and paths for your cluster:

```bash
ROOT=/path/to/your/workdir   # where cosmos-curate/ and the workspace live

srun --partition=<YOUR_PARTITION> --gpus=2 --ntasks=1 \
     --cpus-per-task=18 --mem=230G --time=01:00:00 \
     --account=<YOUR_ACCOUNT> --pty \
apptainer exec --nv --no-mount hostfs,tmp \
  --bind $ROOT/cosmos_curate_local_workspace:/config \
  --bind $ROOT/.config/cosmos_curate/config.yaml:/cosmos_curate/config/cosmos_curate.yaml \
  --bind $ROOT/cosmos_curate_local_workspace/tmp_home:/home/$USER \
  --bind $ROOT/cosmos_curate_local_workspace/tmp:/tmp \
  --bind $ROOT/input_videos:/config/input_videos \
  --bind $ROOT/cosmos-curate/patches/monitoring_patched.py:/opt/cosmos-curate/.pixi/envs/default/lib/python3.12/site-packages/cosmos_xenna/pipelines/private/monitoring.py \
  --bind $ROOT/cosmos-curate/cosmos_curate:/opt/cosmos-curate/cosmos_curate \
  --env TMPDIR=/tmp \
  --env HF_HOME=/config/default_workspace/weights/hf_home/ \
  --env RAY_TMPDIR=/tmp \
  --env COSMOS_CURATOR_LOCAL_DOCKER_JOB=1 \
  --pwd /opt/cosmos-curate \
  $ROOT/cosmos-curate.sif \
  pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
    --input-video-path /config/input_videos \
    --output-clip-path /config/output_clips \
    --embedding-algorithm cradio \
    --limit 5
```

For a **batch job**, see `run_split_annotate.sbatch` at the repo root — fill in the `CONFIGURATION` section at the top.

---

## 9. STREAMING vs BATCH execution mode

cosmos-xenna automatically selects the execution mode:

- **STREAMING** — all pipeline stages run simultaneously (like a conveyor belt). ~40% faster end-to-end. Requires: total GPU requested by all stages ≤ GPUs available.
- **BATCH** — stages execute sequentially (one finishes all videos, then the next starts). Uses less peak memory.

GPU budget for the standard C-RADIO + Qwen + Judge pipeline:

| Stage | GPU |
|---|---|
| TransNetV2ClipExtractionStage | 0.25 |
| CRadioEmbeddingStage | 0.25 |
| VllmCaptionStage | 1.0 |
| JudgeStage | 0.5–1.0 |
| VideoFrameExtractionStage (ffmpeg_gpu) | 0.1 |
| **Total** | **~2.1** |

Use `--gpus=3` to get STREAMING mode with the full judge pipeline. With `--gpus=2`, vllm + judge exceed the available budget and the pipeline falls back to BATCH.

---

## 10. Memory requirements

| Configuration | Minimum RAM | Notes |
|---|---|---|
| C-RADIO + no captions | 50 GB | Embeddings only |
| C-RADIO + Qwen captions (no judge) | 150 GB | VllmPrepStage materialises tokenised frame tensors |
| C-RADIO + Qwen + VCInspector-7B judge | 230 GB | Judge loads additional 16 GB weights |
| Long videos (>5 min each) | 350 GB | VllmPrepStage spawns more workers, each holds tensors |

The `--mem=230G` minimum is not obvious — Ray pre-allocates ~30% of visible RAM as a lazy object store, and VllmPrepStage is the first stage to commit physical pages by writing large tokenised tensors. At `--mem=100G` the job hits OOM at VllmPrepStage.

---

## 11. Downloading model weights

Models are downloaded once and stored in the workspace. Use the quota-monitored download script if your cluster enforces inode/space quotas:

```bash
# Download a specific model (run inside the container or with conda env active)
pixi run python -m cosmos_curate.client.local.local_launcher download \
  --model transnetv2 \
  --workspace ./cosmos_curate_local_workspace
```

Or download directly with `huggingface-cli`:
```bash
huggingface-cli download nvidia/C-RADIOv4-H \
  --local-dir ./cosmos_curate_local_workspace/models/nvidia/C-RADIOv4-H
```

Required models for the full C-RADIO + Qwen + VCInspector pipeline:

| Model | Size | Purpose |
|---|---|---|
| `Sn4kehead/TransNetV2` | 30 MB | Shot detection |
| `nvidia/C-RADIOv4-H` | 2.5 GB | Embedding |
| `google-bert/bert-large-uncased` | 1.3 GB | BERT (InternVideo2 text tower) |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 16 GB | Captioning |
| `dipta007/VCInspector-7B` | 16 GB | Judge |
