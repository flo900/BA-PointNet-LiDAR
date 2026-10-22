#!/bin/bash
#SBATCH --job-name=pointnet_rgb_v1
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/cfs/earth/scratch/troxlflo/BA
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_rgb_v1_%j.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/logs/pointnet_rgb_v1_%j.err

set -e
source /net/home/troxlflo/BA/Projektarbeit-2/scripts/setup_env.sh
conda activate /cfs/earth/scratch/troxlflo/.conda/envs/ba_pointnet

if [ "$#" -ne 0 ]; then
    echo "This job starts a new RGB-v1 run; no baseline resume." >&2
    exit 1
fi

POINTNET_RGB_RUN="/cfs/earth/scratch/troxlflo/BA/runs/train_split_B_rgb_v1_${SLURM_JOB_ID}"
echo "RGB output: ${POINTNET_RGB_RUN}"
python -u /net/home/troxlflo/BA/Projektarbeit-2/scripts/train_pointnet_rgb_v1.py \
    --data-dir /cfs/earth/scratch/troxlflo/BA/data/processed/waedenswil_pointnet_split_B_rgb_v1 \
    --baseline-run-dir /cfs/earth/scratch/troxlflo/BA/runs/train_split_B_b24_1065610 \
    --output-dir "${POINTNET_RGB_RUN}" \
    --epochs 50 --batch-size 24 --num-workers 2 --patience 10 --seed 42

