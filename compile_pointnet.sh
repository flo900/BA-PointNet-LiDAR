#!/bin/bash
#SBATCH --job-name=pointnet_compile
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/pointnet_build/compile.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/pointnet_build/compile.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail

# Cluster-Umgebung laden
module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

# Unsere Python-Umgebung aktivieren
conda activate ba_pointnet

# PointNet++ für die A100 (Compute Capability 8.0) bauen
export TORCH_CUDA_ARCH_LIST="8.0"

BUILD_DIR="/cfs/earth/scratch/troxlflo/BA/pointnet_build/source"
SOURCE_DIR="/net/home/troxlflo/BA/Projektarbeit-2/src/PointnetPP/PointNet2_PyTorch/pointnet2_ops_lib"

echo "=== Umgebung ==="
which python
python --version
nvcc --version

echo "=== PyTorch ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"

# Saubere Kopie des Quellcodes auf Scratch erstellen
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp -a "$SOURCE_DIR"/. "$BUILD_DIR"/

cd "$BUILD_DIR"

echo "=== PointNet++ wird kompiliert ==="

python -m pip install -v --no-build-isolation .

echo "=== Import-Test ==="

python -c "import pointnet2_ops._ext; print('pointnet2_ops._ext erfolgreich geladen')"

echo "=== CUDA-Funktion testen ==="

python - <<'PY'
import torch
from pointnet2_ops.pointnet2_utils import furthest_point_sample

xyz = torch.rand(1, 128, 3, device="cuda").contiguous()
idx = furthest_point_sample(xyz, 16)

print("Input:", xyz.shape)
print("Output:", idx.shape)
print("Device:", idx.device)
print("Farthest Point Sampling funktioniert!")
PY

echo "=== PointNet++ Build erfolgreich ==="