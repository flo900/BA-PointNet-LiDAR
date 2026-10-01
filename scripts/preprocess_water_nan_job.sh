#!/bin/bash
#SBATCH --job-name=preproc_waedi
#SBATCH --partition=earth-5
#SBATCH --array=1-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/preproc_tile%A_%a.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/preproc_tile%A_%a.err

set -e

source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

echo "======================================"
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Tile-Index: ${SLURM_ARRAY_TASK_ID}"
echo "======================================"

python /net/home/troxlflo/BA/Projektarbeit-2/scripts/preprocess_lidar_water_nan.py \
    --config /net/home/troxlflo/BA/Projektarbeit-2/configs/waedenswil.yaml \
    --tile-index ${SLURM_ARRAY_TASK_ID}

echo "======================================"
echo "Ende: $(date)"
echo "PREPROCESSING TILE ${SLURM_ARRAY_TASK_ID} ERFOLGREICH"
echo "======================================"
