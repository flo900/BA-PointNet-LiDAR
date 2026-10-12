#!/bin/bash
#SBATCH --job-name=pointnet_train_B
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_train_B_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_train_B_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_pointnet

# No argument: complete one epoch, then pause to measure runtime/memory.
# One argument: resume that run directory until 50 epochs or early stopping.
if [ "$#" -gt 1 ]; then
    echo "Usage: sbatch train_pointnet_job.sh [existing_run_directory]" >&2
    exit 1
fi
if [ "$#" -eq 1 ]; then
    POINTNET_RUN_DIR="$1"
    POINTNET_EXTRA=(--resume "${POINTNET_RUN_DIR}/last_checkpoint.pth")
else
    POINTNET_RUN_DIR="/cfs/earth/scratch/troxlflo/BA/runs/train_split_B_${SLURM_JOB_ID}"
    POINTNET_EXTRA=(--stop-after 1)
fi
echo "Output: ${POINTNET_RUN_DIR}"
python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/train_pointnet.py \
    --data-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_v1 \
    --output-dir "$POINTNET_RUN_DIR" \
    --epochs 50 --batch-size 8 --num-workers 2 --patience 10 \
    "${POINTNET_EXTRA[@]}"
