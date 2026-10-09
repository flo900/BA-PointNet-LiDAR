#!/bin/bash
#SBATCH --job-name=split_waedi
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/split_waedi_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/split_waedi_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/split_pointnet_dataset.py \
    --input-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_v1 \
    --output-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_v1
