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
module load cuda/12.1
module load anaconda3

# Activate conda env (adjust name as needed)
conda activate yolo

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"
DATASET_YAML="${SCRIPT_DIR}/bus2d_dataset.yaml"

# Update dataset YAML path for HPC
export BUS2D_ROOT="/scratch/${USER}/bus2d/slices"

# ── Launch training ─────────────────────────────────────────────────
# ultralytics handles DDP internally when device=[0,1,2,3]
# No need for torchrun — the YOLO API does it automatically.
echo "Starting BUS-2D training on $(hostname)"
echo "GPUs: $(nvidia-smi -L)"
echo "Config: ${CONFIG}"
echo "Dataset: ${DATASET_YAML}"
echo ""

python "${SCRIPT_DIR}/scripts/train.py" \
    --config "${CONFIG}" \
    --dataset-yaml "${DATASET_YAML}"

echo ""
echo "Training finished at $(date)"
