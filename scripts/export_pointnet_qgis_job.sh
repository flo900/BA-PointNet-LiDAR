#!/bin/bash
#SBATCH --job-name=pointnet_qgis
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_qgis_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_qgis_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_pointnet
python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/export_pointnet_qgis.py \
    --data-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_v1 \
    --checkpoint /cfs/earth/scratch/troxlflo/BA/runs/train_split_B_1065044/best_checkpoint.pth \
    --output-dir "/cfs/earth/scratch/troxlflo/BA/data/qa/pointnet_qgis_${SLURM_JOB_ID}" \
    --batch-size 24
