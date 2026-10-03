#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# ScanNet-SG single-scene experiment runner
#
# Usage:
#   ./run_scannet_sg.sh SCENE RUN_NAME [START_STAGE]
#
# Examples:
#   ./run_scannet_sg.sh scene0000_00 baseline
#   ./run_scannet_sg.sh scene0000_00 baseline sam
#   ./run_scannet_sg.sh scene0000_00 baseline fusion
#
# START_STAGE:
#   ram | sam | fusion | graph | clean
#
# Environment parameters:
#   FRAME_STEP=3
#   CONFIDENCE_THRESHOLD=0.4
#   MAX_DEPTH=0.0
#   SUBSAMPLE_FACTOR=1
#   EDGE_THRESHOLD=2.0
#   VOCAB=/path/to/vocabulary.json
# ============================================================


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------

if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "  $0 SCENE RUN_NAME [START_STAGE]"
    echo
    echo "Examples:"
    echo "  $0 scene0000_00 baseline"
    echo "  $0 scene0000_00 baseline sam"
    echo "  $0 scene0000_00 baseline fusion"
    echo "  $0 scene0000_00 baseline graph"
    exit 1
fi

SCENE="$1"
RUN_NAME="$2"
START_STAGE="${3:-ram}"

case "$START_STAGE" in
    ram|sam|fusion|graph|clean)
        ;;
    *)
        echo "ERROR: START_STAGE must be:"
        echo "  ram | sam | fusion | graph | clean"
        exit 1
        ;;
esac


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${BUILD_DIR:-$REPO/scannet/build-partaware}"

# Prepared RGB-D + pose data
IMAGE_ROOT="${IMAGE_ROOT:-$HOME/datasets/scannet/images/scans}"

# All experiment outputs
PROCESSED_BASE="${PROCESSED_BASE:-$HOME/datasets/scannet-sg-processed}"

# This experiment
RUN_ROOT="$PROCESSED_BASE/$RUN_NAME/openset_scans"

# This scene's final/intermediate outputs
SCENE_OUTPUT="$RUN_ROOT/$SCENE"

REFINED="$SCENE_OUTPUT/refined_instance"

# Experiment-specific RGB-only RAM input
RAM_INPUT_ROOT="$PROCESSED_BASE/$RUN_NAME/ram_input/scans"

RAM_WEIGHT="$HOME/models/ram/ram_plus_swin_large_14m.pth"


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

FRAME_STEP="${FRAME_STEP:-3}"

CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.4}"
GROUNDING_BACKEND="${GROUNDING_BACKEND:-florence}"
case "$GROUNDING_BACKEND" in
    florence|dino) ;;
    *) echo "ERROR: GROUNDING_BACKEND must be florence or dino"; exit 1 ;;
esac

MAX_DEPTH="${MAX_DEPTH:-0.0}"

SUBSAMPLE_FACTOR="${SUBSAMPLE_FACTOR:-1}"

EDGE_THRESHOLD="${EDGE_THRESHOLD:-2.0}"

VOCAB="${VOCAB:-$REPO/scannet/script/ram/scannet_smoke_12.json}"


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

stage_number() {
    case "$1" in
        ram)     echo 1 ;;
        sam)     echo 2 ;;
        fusion)  echo 3 ;;
        graph)   echo 4 ;;
        clean)   echo 5 ;;
    esac
}

START_NUM="$(stage_number "$START_STAGE")"


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

TORCH_LIB="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"

if [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
fi


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo " ScanNet-SG Experiment"
echo "============================================================"
echo "Scene:                $SCENE"
echo "Run name:             $RUN_NAME"
echo "Start stage:          $START_STAGE"
echo
echo "Frame step:           $FRAME_STEP"
echo "Confidence threshold: $CONFIDENCE_THRESHOLD"
echo "Grounding backend:     $GROUNDING_BACKEND"
echo "Max depth:            $MAX_DEPTH"
echo "Pixel subsample:      $SUBSAMPLE_FACTOR"
echo "Edge threshold:       $EDGE_THRESHOLD"
echo
echo "Vocabulary:"
echo "  $VOCAB"
echo
echo "Output:"
echo "  $SCENE_OUTPUT"
echo "============================================================"
echo


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

PREPARED_SCENE="$IMAGE_ROOT/$SCENE"

if [ ! -d "$PREPARED_SCENE" ]; then
    echo "ERROR: prepared scene does not exist:"
    echo "  $PREPARED_SCENE"
    echo
    echo "Run:"
    echo "  python prepare_scannet_scene.py $SCENE"
    exit 1
fi

if [ ! -f "$PREPARED_SCENE/_info.txt" ]; then
    echo "ERROR: _info.txt missing:"
    echo "  $PREPARED_SCENE/_info.txt"
    exit 1
fi

if [ ! -f "$VOCAB" ]; then
    echo "ERROR: vocabulary does not exist:"
    echo "  $VOCAB"
    exit 1
fi

if [ "$START_NUM" -le 1 ] && [ ! -f "$RAM_WEIGHT" ]; then
    echo "ERROR: RAM++ weight does not exist:"
    echo "  $RAM_WEIGHT"
    exit 1
fi

if [ ! -x "$BUILD_DIR/openset_ply_map" ]; then
    echo "ERROR: openset_ply_map not built."
    exit 1
fi

if [ ! -x "$BUILD_DIR/generate_json" ]; then
    echo "ERROR: generate_json not built."
    exit 1
fi


# ------------------------------------------------------------
# Create directories
# ------------------------------------------------------------

mkdir -p "$RAM_INPUT_ROOT/$SCENE"
mkdir -p "$REFINED"


# ------------------------------------------------------------
# RGB-only experiment input
# ------------------------------------------------------------

MAX_FRAMES="${MAX_FRAMES:-0}"

echo "========== Preparing RGB view =========="
echo "Frame step: $FRAME_STEP"

if [ "$MAX_FRAMES" -eq 0 ]; then
    echo "Max frames: unlimited"
else
    echo "Max frames: $MAX_FRAMES"
fi

RAM_SCENE="$RAM_INPUT_ROOT/$SCENE"

# Rebuild this experiment's RGB input.
rm -rf "$RAM_SCENE"
mkdir -p "$RAM_SCENE"

SELECTED=0

for src in "$PREPARED_SCENE"/frame-*.color.jpg; do

    [ -e "$src" ] || continue

    name="$(basename "$src")"

    # frame-000123.color.jpg -> 000123
    frame_string="${name#frame-}"
    frame_string="${frame_string%.color.jpg}"

    # Force decimal interpretation so 000009 is not treated as octal.
    frame_id=$((10#$frame_string))

    # Temporal sampling.
    if (( frame_id % FRAME_STEP != 0 )); then
        continue
    fi

    ln "$src" "$RAM_SCENE/$name"

    SELECTED=$((SELECTED + 1))

    if [ "$MAX_FRAMES" -gt 0 ] && \
       [ "$SELECTED" -ge "$MAX_FRAMES" ]; then
        break
    fi

done

echo "Selected RGB frames: $SELECTED"

if [ "$SELECTED" -eq 0 ]; then
    echo "ERROR: no RGB frames selected."
    exit 1
fi

# ============================================================
# STAGE 1 — RAM++
# ============================================================

if [ "$START_NUM" -le 1 ]; then

    echo
    echo "========== [1/5] RAM++ =========="

    # Important:
    # We intentionally clear only RAM/SAM per-frame files inside
    # THIS experiment before starting RAM again.
    #
    # This prevents an old FRAME_STEP from contaminating a new run.
    #
    # Other RUN_NAME directories are untouched.

    find "$REFINED" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.png' \) \
        -delete

    python "$REPO/scannet/script/ram/inference_ram_given_folders.py" \
        --scans_folder "$RAM_INPUT_ROOT" \
        --output_json_folder "$RUN_ROOT" \
        --start_scene_id "$((10#${SCENE:5:4}))" \
        --end_scene_id "$((10#${SCENE:5:4}))" \
        --pretrained "$RAM_WEIGHT" \
        --process_every_n_images 1\
        --llm_tag_des "$VOCAB"\
        --save_json True

    RAM_JSON_COUNT="$(
        find "$REFINED" -maxdepth 1 -type f \
            -name '*.json' \
            ! -name '*_instance.json' \
            ! -name '*_updated_instance.json' \
            | wc -l
    )"

    echo "RAM JSON files: $RAM_JSON_COUNT"

    if [ "$RAM_JSON_COUNT" -eq 0 ]; then
        echo "ERROR: RAM generated no JSON files."
        exit 1
    fi

else

    echo
    echo "========== [1/5] RAM++ SKIPPED =========="

fi


# ============================================================
# STAGE 2 — GroundingDINO + SAM
# ============================================================

if [ "$START_NUM" -le 2 ]; then

    echo
    echo "========== [2/5] Grounded-SAM =========="

    # If we deliberately restart from SAM, remove only outputs
    # produced by SAM / later per-frame fusion.
    #
    # Original RAM JSON remains.

    if [ "$START_STAGE" = "sam" ]; then

        find "$REFINED" -maxdepth 1 -type f \
            \( \
                -name '*.png' \
                -o -name '*_instance.json' \
                -o -name '*_updated_instance.json' \
            \) \
            -delete

    fi

    RAM_JSON_COUNT="$(
        find "$REFINED" -maxdepth 1 -type f \
            -name '*.json' \
            ! -name '*_instance.json' \
            ! -name '*_updated_instance.json' \
            | wc -l
    )"

    if [ "$RAM_JSON_COUNT" -eq 0 ]; then
        echo "ERROR: no RAM JSON found."
        echo "Cannot start Grounded-SAM."
        exit 1
    fi

    python "$REPO/scannet/script/grounded_sam/scannet_process/get_seg_openset.py" \
        --image_folder "$RAM_INPUT_ROOT/$SCENE" \
        --json_folder "$REFINED" \
        --confidence_threshold "$CONFIDENCE_THRESHOLD" \
        --grounding_backend "$GROUNDING_BACKEND"

    MASK_COUNT="$(
        find "$REFINED" -maxdepth 1 -type f \
            -name '*.png' \
            | wc -l
    )"

    INSTANCE_JSON_COUNT="$(
        find "$REFINED" -maxdepth 1 -type f \
            -name '*_instance.json' \
            ! -name '*_updated_instance.json' \
            | wc -l
    )"

    echo "Masks:         $MASK_COUNT"
    echo "Instance JSON: $INSTANCE_JSON_COUNT"

    if [ "$MASK_COUNT" -eq 0 ]; then
        echo "ERROR: Grounded-SAM generated no masks."
        exit 1
    fi

else

    echo
    echo "========== [2/5] Grounded-SAM SKIPPED =========="

fi


# ============================================================
# STAGE 3 — 3D instance fusion
# ============================================================

if [ "$START_NUM" -le 3 ]; then

    echo
    echo "========== [3/5] 3D Fusion =========="

    MASK_COUNT="$(
        find "$REFINED" -maxdepth 1 -type f \
            -name '*.png' \
            | wc -l
    )"

    if [ "$MASK_COUNT" -eq 0 ]; then
        echo "ERROR: no SAM masks found."
        echo "Cannot start 3D fusion."
        exit 1
    fi

    # Remove old scene-level outputs from THIS run only.
    rm -f \
        "$SCENE_OUTPUT/instance_cloud.ply" \
        "$SCENE_OUTPUT/instance_cloud_with_background.ply" \
        "$SCENE_OUTPUT/colored_instances.ply" \
        "$SCENE_OUTPUT/averaged_instance_features.json" \
        "$SCENE_OUTPUT/instance_bert_embeddings.json" \
        "$SCENE_OUTPUT/instance_name_map.csv" \
        "$SCENE_OUTPUT/topology_map.json" \
        "$SCENE_OUTPUT/instance_cloud_filtered.ply" \
        "$SCENE_OUTPUT/topology_map_filtered.json" \
        "$SCENE_OUTPUT/instance_cloud_cleaned.ply" \
        "$SCENE_OUTPUT/topology_map_original.json"

    # Old updated-instance JSON belongs to old fusion results.
    find "$REFINED" -maxdepth 1 -type f \
        -name '*_updated_instance.json' \
        -delete

    cd "$BUILD_DIR"

    ./openset_ply_map \
        "$SCENE" \
        0 \
        "$RUN_ROOT" \
        "$IMAGE_ROOT" \
        "$MAX_DEPTH" \
        "$SUBSAMPLE_FACTOR"

    cd "$REPO"

    if [ ! -f "$SCENE_OUTPUT/instance_cloud.ply" ]; then
        echo "ERROR: instance_cloud.ply was not generated."
        exit 1
    fi

else

    echo
    echo "========== [3/5] 3D Fusion SKIPPED =========="

fi


# ============================================================
# STAGE 4 — topology_map.json
# ============================================================

if [ "$START_NUM" -le 4 ]; then

    echo
    echo "========== [4/5] Scene Graph =========="

    INSTANCE_CLOUD="$SCENE_OUTPUT/instance_cloud.ply"

    if [ ! -f "$INSTANCE_CLOUD" ]; then
        echo "ERROR: instance_cloud.ply does not exist."
        echo "Cannot generate scene graph."
        exit 1
    fi

    rm -f "$SCENE_OUTPUT/topology_map.json"

    cd "$BUILD_DIR"

    ./generate_json \
        "$INSTANCE_CLOUD" \
        0 \
        1 \
        "$EDGE_THRESHOLD"

    cd "$REPO"

    if [ ! -f "$SCENE_OUTPUT/topology_map.json" ]; then
        echo "ERROR: topology_map.json was not generated."
        exit 1
    fi

else

    echo
    echo "========== [4/5] Scene Graph SKIPPED =========="

fi


# ============================================================
# STAGE 5 — cleaned instance cloud
# Keeps BOTH instance_cloud.ply and instance_cloud_cleaned.ply
# Also preserves the pre-clean graph as topology_map_original.json
# ============================================================

if [ "$START_NUM" -le 5 ]; then

    echo
    echo "========== [5/5] Clean Instance Cloud =========="

    if [ ! -f "$SCENE_OUTPUT/instance_cloud.ply" ]; then
        echo "ERROR: instance_cloud.ply does not exist."
        echo "Cannot generate cleaned point cloud."
        exit 1
    fi

    if [ ! -f "$SCENE_OUTPUT/topology_map.json" ]; then
        echo "ERROR: topology_map.json does not exist."
        echo "Cannot post-filter scene graph."
        exit 1
    fi

    if ! python -c "import open3d" >/dev/null 2>&1; then
        echo "ERROR: Python package open3d is not installed in the current environment."
        echo "Install it with:"
        echo "  pip install open3d"
        exit 1
    fi

    # Preserve the original, pre-clean topology map.
    cp -f "$SCENE_OUTPUT/topology_map.json" \
          "$SCENE_OUTPUT/topology_map_original.json"

    # Remove only the previous cleaned output; keep instance_cloud.ply intact.
    rm -f "$SCENE_OUTPUT/instance_cloud_cleaned.ply"

    # map_ply_post_filter.py expects a real root directory containing scene folders.
    # Do not use a symlink-only temporary root: its recursive file discovery may skip it.
    python "$REPO/scannet/script/map_ply_post_filter.py" \
        "$RUN_ROOT" \
        --openset

    if [ ! -f "$SCENE_OUTPUT/instance_cloud_cleaned.ply" ]; then
        echo "ERROR: instance_cloud_cleaned.ply was not generated."
        exit 1
    fi

    echo "Original PLY kept:"
    echo "  $SCENE_OUTPUT/instance_cloud.ply"
    echo "Cleaned PLY generated:"
    echo "  $SCENE_OUTPUT/instance_cloud_cleaned.ply"
    echo "Original topology map preserved:"
    echo "  $SCENE_OUTPUT/topology_map_original.json"
    echo "Post-filtered topology map:"
    echo "  $SCENE_OUTPUT/topology_map.json"

else

    echo
    echo "========== [5/5] Clean Instance Cloud SKIPPED =========="

fi


# ------------------------------------------------------------
# Finish
# ------------------------------------------------------------

echo
echo "============================================================"
echo " ScanNet-SG SUCCESS"
echo "============================================================"
echo
echo "Scene:"
echo "  $SCENE"
echo
echo "Experiment:"
echo "  $RUN_NAME"
echo
echo "Output directory:"
echo "  $SCENE_OUTPUT"
echo
echo "Scene graph:"
echo "  $SCENE_OUTPUT/topology_map.json"
echo
echo "Original 3D instance cloud:"
echo "  $SCENE_OUTPUT/instance_cloud.ply"
echo
echo "Cleaned 3D instance cloud:"
echo "  $SCENE_OUTPUT/instance_cloud_cleaned.ply"
echo
echo "Original topology map:"
echo "  $SCENE_OUTPUT/topology_map_original.json"
echo
echo "Intermediate files were preserved."
echo "============================================================"
