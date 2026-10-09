#!/bin/bash
#SBATCH --job-name=sample_25m
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/sample_25m_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/sample_25m_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/create_pointnet_dataset.py \
    --config /net/home/troxlflo/BA/Projektarbeit-2/configs/waedenswil.yaml \
    --output /cfs/earth/scratch/troxlflo/BA/data/qa/tiles_25m_sampled_inventory.csv \
    --sample-dir /cfs/earth/scratch/troxlflo/BA/data/intermediate/waedenswil_sampled_25m_v1 \
    --min-points 4000
