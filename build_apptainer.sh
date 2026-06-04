#!/bin/bash
# Build the Cosmos-Curate Apptainer SIF image.
#
# Usage:
#   export APPTAINER_DOCKER_USERNAME='$oauthtoken'
#   export APPTAINER_DOCKER_PASSWORD='<your-NGC-API-key>'
#   bash build_apptainer.sh
#
# The resulting image is written to <your-project-dir>/cosmos-curate.sif
# Set PROJECT_DIR below before running.
#
# Safety: monitors home directory quota and kills the build if it increases
# by more than 0.5%, to protect against unexpected writes to $HOME.
# Note: quota monitoring uses the `myquota` command (Snellius-specific).
# Remove or replace get_home_quota_pct / get_work4_inodes_pct if your cluster
# uses a different quota tool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"   # override with: PROJECT_DIR=/your/path bash build_apptainer.sh
SIF_PATH="${PROJECT_DIR}/cosmos-curate.sif"
DEF_PATH="${SCRIPT_DIR}/cosmos-curate.def"
MAX_HOME_QUOTA_INCREASE="0.5"
MAX_WORK4_INODES_PCT="96.5"

# Verify NGC credentials are set (needed to pull from nvcr.io)
if [ -z "${APPTAINER_DOCKER_PASSWORD:-}" ]; then
    echo "ERROR: APPTAINER_DOCKER_PASSWORD is not set."
    echo "  export APPTAINER_DOCKER_USERNAME='\$oauthtoken'"
    echo "  export APPTAINER_DOCKER_PASSWORD='<your-NGC-API-key>'"
    exit 1
fi

# ----------------------------------------------------------------
# Helper: read current home quota usage percentage
# ----------------------------------------------------------------
get_home_quota_pct() {
    # Snellius: myquota <project> | grep home line
    # Replace with your cluster's quota command if different
    myquota "${QUOTA_PROJECT:-}" 2>/dev/null \
        | grep -A 3 "home" \
        | grep "GiB" \
        | grep -oP '\d+\.\d+(?=%)' \
        | head -1
}

get_work4_inodes_pct() {
    myquota "${QUOTA_PROJECT:-}" 2>/dev/null \
        | grep -A 4 "wstor_work4\|scratch\|work" \
        | grep "Inodes" \
        | grep -oP '\d+\.\d+(?=%)' \
        | head -1
}

# ----------------------------------------------------------------
# Capture baseline home quota
# ----------------------------------------------------------------
BASELINE_QUOTA=$(get_home_quota_pct)
if [ -z "${BASELINE_QUOTA}" ]; then
    echo "ERROR: Could not read home directory quota. Aborting."
    exit 1
fi
BASELINE_WORK4_INODES=$(get_work4_inodes_pct)
echo "[quota-guard] Baseline home quota: ${BASELINE_QUOTA}%"
echo "[quota-guard] Will kill build if home increase exceeds ${MAX_HOME_QUOTA_INCREASE}%"
echo "[quota-guard] Baseline work4 inodes: ${BASELINE_WORK4_INODES}%"
echo "[quota-guard] Will kill build if work4 inodes exceed ${MAX_WORK4_INODES_PCT}%"
echo ""

# ----------------------------------------------------------------
# Redirect all Apptainer dirs into the project to avoid $HOME writes
# ----------------------------------------------------------------
export APPTAINER_TMPDIR="${PROJECT_DIR}/.apptainer_tmp"
export APPTAINER_CACHEDIR="${PROJECT_DIR}/.apptainer_cache"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

echo "Building Apptainer image..."
echo "  Definition:  ${DEF_PATH}"
echo "  Output:      ${SIF_PATH}"
echo "  TMPDIR:      ${APPTAINER_TMPDIR}"
echo "  CACHEDIR:    ${APPTAINER_CACHEDIR}"
echo ""

# ----------------------------------------------------------------
# Background quota monitor
# ----------------------------------------------------------------
MONITOR_PID=""

quota_monitor() {
    local baseline="$1"
    local build_pid="$2"
    local threshold="$3"
    local inodes_limit="$4"

    while kill -0 "${build_pid}" 2>/dev/null; do
        sleep 30

        current=$(get_home_quota_pct)
        if [ -z "${current}" ]; then
            echo "[quota-guard] WARNING: could not read quota, retrying..."
            continue
        fi

        # Calculate home quota increase
        increase=$(awk "BEGIN { printf \"%.4f\", ${current} - ${baseline} }")
        exceeded=$(awk "BEGIN { print (${increase} > ${threshold}) ? 1 : 0 }")

        # Check work4 inodes
        current_inodes=$(get_work4_inodes_pct)
        inodes_exceeded=$(awk "BEGIN { print (\"${current_inodes}\" != \"\" && ${current_inodes} > ${inodes_limit}) ? 1 : 0 }")

        echo "[quota-guard] Home quota: ${current}% (baseline ${baseline}%, delta +${increase}%)  |  work4 inodes: ${current_inodes}%"

        if [ "${exceeded}" -eq 1 ]; then
            echo ""
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "[quota-guard] ALERT: Home quota increased by +${increase}%"
            echo "[quota-guard]   Baseline: ${baseline}%  Current: ${current}%"
            echo "[quota-guard]   Threshold: +${threshold}%"
            echo "[quota-guard] KILLING apptainer build (PID ${build_pid}) to protect home directory!"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo ""
            kill -TERM "${build_pid}" 2>/dev/null
            sleep 2
            kill -KILL "${build_pid}" 2>/dev/null
            exit 1
        fi

        if [ "${inodes_exceeded}" -eq 1 ]; then
            echo ""
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo "[quota-guard] ALERT: work4 inodes at ${current_inodes}% (limit ${inodes_limit}%)"
            echo "[quota-guard] KILLING apptainer build (PID ${build_pid}) to protect inode quota!"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            echo ""
            kill -TERM "${build_pid}" 2>/dev/null
            sleep 2
            kill -KILL "${build_pid}" 2>/dev/null
            exit 1
        fi
    done
    echo "[quota-guard] Build process finished, stopping monitor."
}

# ----------------------------------------------------------------
# Cleanup: always stop the monitor and report final quota
# ----------------------------------------------------------------
cleanup() {
    if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
    final_quota=$(get_home_quota_pct)
    final_inodes=$(get_work4_inodes_pct)
    echo ""
    echo "[quota-guard] Final home quota:   ${final_quota:-unknown}% (baseline was ${BASELINE_QUOTA}%)"
    echo "[quota-guard] Final work4 inodes: ${final_inodes:-unknown}% (limit was ${MAX_WORK4_INODES_PCT}%)"
}
trap cleanup EXIT

# ----------------------------------------------------------------
# Launch build + monitor
# ----------------------------------------------------------------
cd "${SCRIPT_DIR}"

apptainer build --fakeroot --force "${SIF_PATH}" "${DEF_PATH}" &
BUILD_PID=$!

quota_monitor "${BASELINE_QUOTA}" "${BUILD_PID}" "${MAX_HOME_QUOTA_INCREASE}" "${MAX_WORK4_INODES_PCT}" &
MONITOR_PID=$!

# Wait for the build to finish
wait "${BUILD_PID}"
BUILD_EXIT=$?

# Give monitor a moment to print its final message, then stop it
sleep 2
if kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null
    wait "${MONITOR_PID}" 2>/dev/null || true
fi

if [ "${BUILD_EXIT}" -ne 0 ]; then
    echo ""
    echo "ERROR: apptainer build failed with exit code ${BUILD_EXIT}"
    exit "${BUILD_EXIT}"
fi

echo ""
echo "Build complete: ${SIF_PATH}"
echo "Image size: $(du -sh "${SIF_PATH}" | cut -f1)"
