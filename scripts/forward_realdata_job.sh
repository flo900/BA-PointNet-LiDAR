#!/bin/bash

#SBATCH --job-name=pointnet_forward
#SBATCH --partition=earth-5
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/pointnet_forward_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/pointnet_forward_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA

# --------------------------------------------------
# In unser Repository wechseln
# --------------------------------------------------

cd /net/home/troxlflo/BA/Projektarbeit-2

# --------------------------------------------------
# PointNet-Umgebung laden
# --------------------------------------------------

source scripts/setup_env.sh

# --------------------------------------------------
# Echtes LiDAR-Tile durch PointNet++ schicken
# --------------------------------------------------

python scripts/test_pointnet_forward_realdata.py
