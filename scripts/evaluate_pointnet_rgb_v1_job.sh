#!/bin/bash
#SBATCH --job-name=test_rgb_v1
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_rgb_v1_test_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_rgb_v1_test_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_pointnet

python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/evaluate_pointnet_rgb_v1.py \
    --data-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_rgb_v1 \
    --checkpoint /cfs/earth/scratch/troxlflo/BA/runs/train_split_B_rgb_v1_1073682/best_checkpoint.pth \
    --output-dir "/cfs/earth/scratch/troxlflo/BA/runs/test_split_B_rgb_v1_${SLURM_JOB_ID}" \
    --expected-epoch 48 --batch-size 24

