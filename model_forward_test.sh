#!/bin/bash
#SBATCH --job-name=model_forward_test
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/gpu_test/model_forward_test.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/gpu_test/model_forward_test.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -euo pipefail

# Cluster-Software laden
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

# Unsere Python-Umgebung
conda activate ba_pointnet

# Damit Python Nicolas Modell-Datei findet
export PYTHONPATH="/net/home/troxlflo/BA/Projektarbeit-2/src/PointnetPP:${PYTHONPATH:-}"

python - <<'PY'

import torch
from pointnet2_aerial_optimized import PointNet2AerialSSG, DiceLoss

print("=== Umgebung ===")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda")

print("\n=== Modell erstellen ===")

model = PointNet2AerialSSG(
    num_classes=5,
    input_channels=4,
    use_xyz=True,
    dropout=0.6
).to(device)

model.eval()

total_params = sum(p.numel() for p in model.parameters())
print("Parameter:", f"{total_params:,}")

print("\n=== Künstlichen LiDAR-Tile erstellen ===")

B = 1
N = 16384

# XYZ-Koordinaten
xyz = torch.rand(B, N, 3, device=device)

# ungefähr auf einen 25 x 25 m Tile skalieren
xyz[:, :, 0] *= 25.0
xyz[:, :, 1] *= 25.0
xyz[:, :, 2] *= 20.0

# 4 zusätzliche LiDAR-Features
extra_features = torch.rand(B, N, 4, device=device)

# XYZ + 4 Features = 7 Werte pro Punkt
dummy_input = torch.cat([xyz, extra_features], dim=2).contiguous()

print("Input shape:", dummy_input.shape)
print("Input device:", dummy_input.device)

print("\n=== Forward Pass ===")

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)
print("Output device:", output.device)

assert output.shape == (B, 5, N)
assert torch.isfinite(output).all()

print("\n=== Klassen ableiten ===")

predictions = output.argmax(dim=1)

print("Prediction shape:", predictions.shape)
print("Vorhandene Klassen:", torch.unique(predictions).tolist())

print("\n=== Dice Loss testen ===")

dummy_labels = torch.randint(
    low=0,
    high=5,
    size=(B, N),
    device=device
)

criterion = DiceLoss()
loss = criterion(output, dummy_labels)

print("Dice Loss:", loss.item())

print("\n=== TEST ERFOLGREICH ===")
print("Nicolas komplettes PointNet++ Modell läuft auf der GPU.")

PY