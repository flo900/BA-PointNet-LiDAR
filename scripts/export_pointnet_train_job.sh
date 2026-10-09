#!/bin/bash
#SBATCH --job-name=export_train
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/export_train_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/export_train_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/export_pointnet_train.py \
    --sample-dir /cfs/earth/scratch/troxlflo/BA/data/intermediate/waedenswil_sampled_25m_v1 \
    --inventory /cfs/earth/scratch/troxlflo/BA/data/qa/tiles_25m_sampled_inventory.csv \
    --output-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_v1 \
    --expected-tiles 6110
