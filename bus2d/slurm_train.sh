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
CONFIG="${SCRIPT_DIR}/config.yaml"
DATASET_YAML="${SCRIPT_DIR}/bus2d_dataset.yaml"

# ── Launch training (reads tar shards directly, no unpacking) ──────
echo "Starting BUS-2D training on $(hostname)"
echo "Shards: ${SHARDS_DIR}"
echo "GPUs: $(nvidia-smi -L)"
echo "Start time: $(date)"
echo ""

python "${SCRIPT_DIR}/scripts/train_wds.py" \
    --config "${CONFIG}" \
    --shards-dir "${SHARDS_DIR}" \
    --dataset-yaml "${DATASET_YAML}"

echo ""
echo "Training finished at $(date)"
