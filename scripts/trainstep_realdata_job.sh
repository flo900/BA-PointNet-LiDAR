#!/bin/bash

#SBATCH --job-name=pointnet_trainstep
#SBATCH --partition=earth-5
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/pointnet_trainstep_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/pointnet_trainstep_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA

cd /net/home/troxlflo/BA/Projektarbeit-2

source scripts/setup_env.sh

python scripts/test_pointnet_trainstep_realdata.py
