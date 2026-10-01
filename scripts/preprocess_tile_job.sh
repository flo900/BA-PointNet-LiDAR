#!/bin/bash
#SBATCH --job-name=preproc_tile0
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/preproc_tile0_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/preproc_tile0_%j.err

set -e

# ZHAW-Module / Conda vorbereiten
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh

# LiDAR-Preprocessing-Environment
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

echo "======================================"
echo "Start: $(date)"
echo "Host:  $(hostname)"
echo "Python: $(which python)"
echo "======================================"

python /net/home/troxlflo/BA/Projektarbeit-2/scripts/preprocess_lidar_refactored.py \
    --config /net/home/troxlflo/BA/Projektarbeit-2/configs/waedenswil.yaml \
    --tile-index 0

echo "======================================"
echo "Ende: $(date)"
echo "PREPROCESSING JOB ERFOLGREICH"
echo "======================================"
