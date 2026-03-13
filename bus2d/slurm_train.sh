#!/bin/bash
#SBATCH --job-name=bus2d-yolo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=nvidia
#SBATCH --output=bus2d_train_%j.log
#SBATCH --error=bus2d_train_%j.err

# ── Environment ─────────────────────────────────────────────────────
module purge
module load cuda/11.8.0
module load anaconda3

conda activate yolo

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARDS_DIR="/scratch/ll5582/Data/Ultrasound/YOLO/shards"
LOCAL_DATA="${TMPDIR:-/tmp}/bus2d_slices"
DATASET_YAML="${SCRIPT_DIR}/bus2d_dataset.yaml"
CONFIG="${SCRIPT_DIR}/config.yaml"

# ── Step 1: Unpack shards to node-local storage ────────────────────
# Uses $TMPDIR (local SSD, no inode quota) — cleaned up when job ends
echo "Unpacking shards to local storage: ${LOCAL_DATA}"
echo "Start time: $(date)"
python "${SCRIPT_DIR}/scripts/unpack_shards.py" \
    --shards-dir "${SHARDS_DIR}" \
    --output-dir "${LOCAL_DATA}"

# Verify
N_TRAIN=$(ls "${LOCAL_DATA}/images/train/" | wc -l)
N_VAL=$(ls "${LOCAL_DATA}/images/val/" | wc -l)
echo "Unpacked: ${N_TRAIN} train, ${N_VAL} val images"

# ── Step 2: Patch dataset YAML with actual local path ──────────────
RUNTIME_YAML="${LOCAL_DATA}/bus2d_dataset.yaml"
sed "s|path:.*|path: ${LOCAL_DATA}|" "${DATASET_YAML}" > "${RUNTIME_YAML}"

# ── Step 3: Launch training ────────────────────────────────────────
echo ""
echo "Starting BUS-2D training on $(hostname)"
echo "GPUs: $(nvidia-smi -L)"
echo ""

python "${SCRIPT_DIR}/scripts/train.py" \
    --config "${CONFIG}" \
    --dataset-yaml "${RUNTIME_YAML}"

echo ""
echo "Training finished at $(date)"

# ── Cleanup ($TMPDIR is auto-cleaned, but be explicit) ─────────────
rm -rf "${LOCAL_DATA}"
echo "Local data cleaned up."
