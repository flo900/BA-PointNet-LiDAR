#!/bin/bash
#SBATCH --job-name=test_water_nan
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/test_water_nan_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/test_water_nan_%j.err

set -e

source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_lidar

python /net/home/troxlflo/BA/Projektarbeit-2/scripts/preprocess_lidar_water_nan.py \
    --config /net/home/troxlflo/BA/Projektarbeit-2/configs/waedenswil_water_nan_test.yaml \
    --tile-index 3 \
    --chunk-index 78
