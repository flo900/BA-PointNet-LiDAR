#!/bin/bash
#SBATCH --job-name=tiles_25m_qa
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/tiles_25m_qa_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/tiles_25m_qa_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/create_pointnet_dataset.py \
    --config /net/home/troxlflo/BA/Projektarbeit-2/configs/waedenswil.yaml \
    --output /cfs/earth/scratch/troxlflo/BA/data/qa/tiles_25m_inventory.csv
