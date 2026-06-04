#!/bin/bash
#
# Cosmos-Curate Split-Annotate Pipeline — Apptainer on Slurm
#
# Usage:
#   Edit the CONFIGURATION section below, then:
#   sbatch run_split_annotate.sbatch
#

# ============================================================
# SLURM CONFIGURATION — edit these for your cluster
# ============================================================
#SBATCH --job-name=cosmos-curate-split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --dependency=singleton
##SBATCH --account=<your_account>
##SBATCH --partition=<your_gpu_partition>
##SBATCH --gres=gpu:4
##SBATCH --time=08:00:00
##SBATCH --output=<your_project_dir>/logs/%x_%j.log

# ============================================================
# PATH CONFIGURATION — edit these for your data
# ============================================================
PROJECT_DIR="<your_project_dir>"   # e.g. /gpfs/work/myproject/cosmos
SIF_IMAGE="${PROJECT_DIR}/cosmos-curate.sif"
WORKSPACE="${PROJECT_DIR}/cosmos_curate_local_workspace"
CONFIG_FILE="${PROJECT_DIR}/.config/cosmos_curate/config.yaml"

# Pipeline input/output (use container-internal paths with /config prefix,
# or S3 paths like s3://bucket/path)
INPUT_VIDEO_PATH="/config/raw_videos"
OUTPUT_CLIP_PATH="/config/output_clips"
LIMIT=1

# ============================================================
# Derived paths (no need to edit)
# ============================================================
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${WORKSPACE}"

# ============================================================
# Job info
# ============================================================
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Nodelist:= ${SLURM_JOB_NODELIST}"
echo "Number of nodes:= ${SLURM_JOB_NUM_NODES}"
echo "Ntasks per node:= ${SLURM_NTASKS_PER_NODE}"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

head_node=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)

# ============================================================
# Environment variables passed into the container
# ============================================================
export COSMOS_S3_PROFILE_PATH="/creds/s3_creds"
export COSMOS_AZURE_PROFILE_PATH="/creds/azure_creds"
export COSMOS_CURATOR_RAY_SLURM_JOB=True
export HEAD_NODE_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * ${SLURM_NTASKS_PER_NODE:-1}))
export HEAD_NODE_ADDR="${head_node}"
export PRIMARY_NODE_HOSTNAME="${head_node}"
export PRIMARY_NODE_PORT="${HEAD_NODE_PORT}"
export TQDM_MININTERVAL=9000

echo "HEAD_NODE_ADDR=${HEAD_NODE_ADDR}"
echo "HEAD_NODE_PORT=${HEAD_NODE_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"

# ============================================================
# Build bind-mount string for Apptainer
# ============================================================
BIND_MOUNTS="${WORKSPACE}:/config"
BIND_MOUNTS="${BIND_MOUNTS},${CONFIG_FILE}:/cosmos_curate/config/cosmos_curate.yaml"

# S3 credentials (optional)
if [ -f "${PROJECT_DIR}/.aws/credentials" ]; then
    BIND_MOUNTS="${BIND_MOUNTS},${PROJECT_DIR}/.aws/credentials:/creds/s3_creds:ro"
fi

# Azure credentials (optional)
if [ -f "${PROJECT_DIR}/.azure/credentials" ]; then
    BIND_MOUNTS="${BIND_MOUNTS},${PROJECT_DIR}/.azure/credentials:/creds/azure_creds:ro"
fi

echo "Bind mounts: ${BIND_MOUNTS}"
echo ""
echo "$(date) - Launching Split-Annotate pipeline..."

# ============================================================
# Run
# ============================================================
srun apptainer exec --nv \
    --writable-tmpfs \
    --no-home \
    --bind "${BIND_MOUNTS}" \
    --pwd /opt/cosmos-curate \
    --env "COSMOS_S3_PROFILE_PATH=${COSMOS_S3_PROFILE_PATH}" \
    --env "COSMOS_AZURE_PROFILE_PATH=${COSMOS_AZURE_PROFILE_PATH}" \
    --env "COSMOS_CURATOR_RAY_SLURM_JOB=${COSMOS_CURATOR_RAY_SLURM_JOB}" \
    --env "HEAD_NODE_ADDR=${HEAD_NODE_ADDR}" \
    --env "HEAD_NODE_PORT=${HEAD_NODE_PORT}" \
    --env "WORLD_SIZE=${WORLD_SIZE}" \
    --env "PRIMARY_NODE_HOSTNAME=${PRIMARY_NODE_HOSTNAME}" \
    --env "PRIMARY_NODE_PORT=${PRIMARY_NODE_PORT}" \
    --env "TQDM_MININTERVAL=${TQDM_MININTERVAL}" \
    "${SIF_IMAGE}" \
    pixi run python -m cosmos_curate.pipelines.video.run_pipeline split \
        --input-video-path "${INPUT_VIDEO_PATH}" \
        --output-clip-path "${OUTPUT_CLIP_PATH}" \
        --limit "${LIMIT}"

echo "$(date) - Pipeline finished."
