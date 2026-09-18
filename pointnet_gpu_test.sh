#!/bin/bash
#SBATCH --job-name=pointnet_test
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/gpu_test/pointnet_test.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/gpu_test/pointnet_test.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

set -euo pipefail

module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

conda activate ba_pointnet

python - <<'PY'
import torch
from pointnet2_ops.pointnet2_utils import furthest_point_sample

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

# künstliche Mini-Punktwolke:
# 1 Tile, 128 Punkte, XYZ
xyz = torch.rand(1, 128, 3, device="cuda").contiguous()

# PointNet++ wählt daraus 16 möglichst gut verteilte Punkte
idx = furthest_point_sample(xyz, 16)

print("Input:", xyz.shape)
print("Output:", idx.shape)
print("Device:", idx.device)
print("Farthest Point Sampling funktioniert!")
PY