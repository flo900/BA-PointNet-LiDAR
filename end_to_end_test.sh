#!/bin/bash
#SBATCH --job-name=pointnet_e2e
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/gpu_test/end_to_end_test.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/gpu_test/end_to_end_test.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -euo pipefail

# Cluster-Umgebung laden
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

# Unsere Python-Umgebung
conda activate ba_pointnet

# Damit Python Nicolas PointNet++-Dateien findet
export PYTHONPATH="/net/home/troxlflo/BA/Projektarbeit-2/src/PointnetPP:${PYTHONPATH:-}"

python - <<'PY'

import torch

from lidar_dataloader import get_dataloaders
from pointnet2_aerial_optimized import PointNet2AerialSSG


DATA_DIR = "/cfs/earth/scratch/troxlflo/BA/dummy_lidar"

device = torch.device("cuda")

print("=== 1. Umgebung ===")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# 2. Nicolas Dataloader verwenden
# ---------------------------------------------------------

print("\n=== 2. Dataloader ===")

train_loader, val_loader, test_loader, metadata = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=1,
    num_workers=0,
    pin_memory=False,
    train_transform=None
)

features, labels = next(iter(train_loader))

print("Geladene Features:", features.shape)
print("Geladene Labels:", labels.shape)


# ---------------------------------------------------------
# 3. Daten auf die GPU verschieben
# ---------------------------------------------------------

print("\n=== 3. Daten -> GPU ===")

features = features.to(device)
labels = labels.to(device)

print("Features device:", features.device)
print("Labels device:", labels.device)


# ---------------------------------------------------------
# 4. Nicolas Modell erstellen
# ---------------------------------------------------------

print("\n=== 4. PointNet++ Modell ===")

model = PointNet2AerialSSG(
    num_classes=5,
    input_channels=4,
    use_xyz=True,
    dropout=0.6
).to(device)

model.eval()

print("Modell erstellt")


# ---------------------------------------------------------
# 5. Forward Pass
# ---------------------------------------------------------

print("\n=== 5. Forward Pass ===")

with torch.no_grad():
    output = model(features)

print("Output:", output.shape)
print("Output device:", output.device)


# ---------------------------------------------------------
# 6. Klassen vorhersagen
# ---------------------------------------------------------

print("\n=== 6. Prediction ===")

predictions = torch.argmax(output, dim=1)

print("Prediction:", predictions.shape)
print("Prediction device:", predictions.device)
print("Vorhergesagte Klassen:", torch.unique(predictions).tolist())


# ---------------------------------------------------------
# 7. Plausibilitätschecks
# ---------------------------------------------------------

assert features.shape == (1, 16384, 7)
assert labels.shape == (1, 16384)
assert output.shape == (1, 5, 16384)
assert predictions.shape == (1, 16384)

assert torch.isfinite(output).all()

print("\n=== END-TO-END TEST ERFOLGREICH ===")
print(".npy -> Dataloader -> GPU -> PointNet++ -> Prediction funktioniert.")

PY