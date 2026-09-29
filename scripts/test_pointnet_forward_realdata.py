import sys
import torch
import numpy as np

sys.path.insert(0, "src/PointnetPP")

from lidar_dataloader import LiDARTileDataset
from pointnet2_aerial_optimized import PointNet2AerialSSG


# --------------------------------------------------
# Daten
# --------------------------------------------------

DATA_DIR = (
    "/cfs/earth/scratch/troxlflo/BA/data/processed/"
    "pointnet_prototype"
)

dataset = LiDARTileDataset(
    data_dir=DATA_DIR,
    split="train"
)

features, labels = dataset[0]

# Batch-Dimension ergänzen:
# (16384, 7) -> (1, 16384, 7)
features = features.unsqueeze(0)

# GPU verwenden
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

features = features.to(device)
labels = labels.to(device)


# --------------------------------------------------
# Modell
# --------------------------------------------------

model = PointNet2AerialSSG(
    num_classes=5,
    input_channels=4,
    dropout=0.6
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# Forward Pass
# --------------------------------------------------

with torch.no_grad():

    output = model(
        features
    )


# --------------------------------------------------
# Kontrolle
# --------------------------------------------------

print()
print("Input shape:")
print(features.shape)

print()
print("Output shape:")
print(output.shape)

predictions = torch.argmax(
    output,
    dim=1
)

print()
print("Predictions shape:")
print(predictions.shape)

print()
print("Vorhergesagte Klassen:")
print(
    torch.unique(
        predictions
    ).cpu().numpy()
)

print()
print("Forward Pass erfolgreich.")
