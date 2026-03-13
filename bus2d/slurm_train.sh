#!/bin/bash
#SBATCH --job-name=bus2d-yolo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=90:00:00
#SBATCH --partition=nvidia
#SBATCH --output=logs/bus2d_train_%j.log
#SBATCH --error=logs/bus2d_train_%j.err

# ── Environment ─────────────────────────────────────────────────────
module purge
module load cuda/11.8.0

conda activate /scratch/ll5582/envs/yolo

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARDS_DIR="/scratch/ll5582/Data/Ultrasound/YOLO/shards"
CONFIG="${SCRIPT_DIR}/config.yaml"
DATASET_YAML="${SCRIPT_DIR}/bus2d_dataset.yaml"

# ── Launch training (1 GPU, reads tar shards directly) ─────────────
echo "Starting BUS-2D training on $(hostname)"
echo "Shards: ${SHARDS_DIR}"
echo "GPU: $(nvidia-smi -L)"
echo "Start time: $(date)"
echo ""

python "${SCRIPT_DIR}/scripts/train_wds.py" \
    --config "${CONFIG}" \
    --shards-dir "${SHARDS_DIR}" \
    --dataset-yaml "${DATASET_YAML}" \
    --device 0

echo ""
echo "Training finished at $(date)"
