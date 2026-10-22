#!/bin/bash
#SBATCH --job-name=export_rgb_v1
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/export_rgb_v1_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/export_rgb_v1_%j.err

set -euo pipefail

# Use the isolated RGB environment directly; no change to baseline setup/envs.
/cfs/earth/scratch/troxlflo/.conda/envs/ba_rgb/bin/python -u \
    /net/home/troxlflo/BA/Projektarbeit-2/scripts/export_pointnet_rgb_v1.py \
    --source-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_v1 \
    --image-dir /cfs/earth/scratch/troxlflo/BA/data/raw/swissimage_2025_rgb \
    --output-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_rgb_v1
