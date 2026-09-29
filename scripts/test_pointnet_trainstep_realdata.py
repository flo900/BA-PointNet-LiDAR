import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src/PointnetPP")

from lidar_dataloader import LiDARTileDataset
from pointnet2_aerial_optimized import PointNet2AerialSSG, DiceLoss


# --------------------------------------------------
# Focal Loss wie bei Nicola
# --------------------------------------------------

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):

        logits = (
            logits.permute(0, 2, 1)
            .contiguous()
            .view(-1, logits.size(1))
        )

        targets = targets.view(-1)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_term = (
            (1 - pt) ** self.gamma
        )

        if self.alpha is not None:
            focal_term = (
                self.alpha[targets]
                * focal_term
            )

        return (
            focal_term * ce_loss
        ).mean()


# --------------------------------------------------
# Daten laden
# --------------------------------------------------

DATA_DIR = (
    "/cfs/earth/scratch/troxlflo/BA/data/processed/"
    "pointnet_prototype"
)

dataset = LiDARTileDataset(
    DATA_DIR,
    split="train"
)

features, labels = dataset[0]

# Batch-Dimension
features = features.unsqueeze(0)
labels = labels.unsqueeze(0)


# --------------------------------------------------
# GPU
# --------------------------------------------------

device = torch.device("cuda")

features = features.to(device)
labels = labels.to(device)

print("Device:", device)


# --------------------------------------------------
# Modell
# --------------------------------------------------

model = PointNet2AerialSSG(
    num_classes=5,
    input_channels=4,
    dropout=0.6
).to(device)

model.train()


# --------------------------------------------------
# Loss-Funktionen
# --------------------------------------------------

class_weights = torch.tensor(
    [2.0, 0.6, 0.2, 0.4, 0.45],
    dtype=torch.float32,
    device=device
)

focal_loss_fn = FocalLoss(
    gamma=2.0,
    alpha=class_weights
)

dice_loss_fn = DiceLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=5e-5
)


# --------------------------------------------------
# Ein Trainingsschritt
# --------------------------------------------------

optimizer.zero_grad()

output = model(features)

focal_loss = focal_loss_fn(
    output,
    labels
)

dice_loss = dice_loss_fn(
    output,
    labels
)

loss = (
    focal_loss
    + 0.2 * dice_loss
)

print()
print("Focal Loss:", focal_loss.item())
print("Dice Loss:", dice_loss.item())
print("Total Loss:", loss.item())

loss.backward()

optimizer.step()


# --------------------------------------------------
# Kontrolle
# --------------------------------------------------

print()
print("Input shape:", features.shape)
print("Output shape:", output.shape)

print()
print("Backward erfolgreich.")
print("Optimizer-Step erfolgreich.")
print("TRAININGSSCHRITT ERFOLGREICH")
