#!/bin/bash
#SBATCH --job-name=qgis_rgb_compare
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/qgis_rgb_compare_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/qgis_rgb_compare_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_pointnet

python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/export_pointnet_rgb_comparison_qgis.py \
    --baseline-data /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_v1 \
    --rgb-data /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_rgb_v1 \
    --baseline-checkpoint /cfs/earth/scratch/troxlflo/BA/runs/train_split_B_b24_1065610/best_checkpoint.pth \
    --rgb-checkpoint /cfs/earth/scratch/troxlflo/BA/runs/train_split_B_rgb_v1_1073682/best_checkpoint.pth \
    --selection-csv /cfs/earth/scratch/troxlflo/BA/data/qa/pointnet_qgis_1065048/selected_tiles.csv \
    --output-dir "/cfs/earth/scratch/troxlflo/BA/data/qa/qgis_rgb_compare_${SLURM_JOB_ID}"
