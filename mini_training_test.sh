#!/bin/bash
#SBATCH --job-name=mini_train_test
#SBATCH --output=/cfs/earth/scratch/troxlflo/BA/gpu_test/mini_train_test.out
#SBATCH --error=/cfs/earth/scratch/troxlflo/BA/gpu_test/mini_train_test.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

set -euo pipefail

module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

conda activate ba_pointnet

export PYTHONPATH="/net/home/troxlflo/BA/Projektarbeit-2/src/PointnetPP:${PYTHONPATH:-}"

python - <<'PY'

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_aerial_optimized import PointNet2AerialSSG, DiceLoss


# ---------------------------------------------------------
# Nicolas Focal Loss
# ---------------------------------------------------------

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):

        # (B, C, N) -> (B*N, C)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits.view(-1, logits.size(-1))

        # (B, N) -> (B*N)
        targets = targets.view(-1)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_term = alpha_t * focal_term

        return (focal_term * ce_loss).mean()


device = torch.device("cuda")

print("=== Umgebung ===")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)


# ---------------------------------------------------------
# 1. Modell erstellen
# ---------------------------------------------------------

model = PointNet2AerialSSG(
    num_classes=5,
    input_channels=4,
    use_xyz=True,
    dropout=0.6
).to(device)

# WICHTIG:
# diesmal Trainingsmodus statt model.eval()
model.train()


# ---------------------------------------------------------
# 2. Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=5e-5
)


# ---------------------------------------------------------
# 3. Loss-Funktionen wie bei Nicola
# ---------------------------------------------------------

class_weights = torch.tensor(
    [2.0, 0.6, 0.2, 0.4, 0.45],
    device=device
)

focal_loss_fn = FocalLoss(
    gamma=2.0,
    alpha=class_weights
)

dice_loss_fn = DiceLoss()


# ---------------------------------------------------------
# 4. Künstliche LiDAR-Daten
# ---------------------------------------------------------

B = 1
N = 16384

xyz = torch.rand(B, N, 3, device=device)

xyz[:, :, 0] *= 25.0
xyz[:, :, 1] *= 25.0
xyz[:, :, 2] *= 20.0

extra_features = torch.rand(
    B, N, 4,
    device=device
)

features = torch.cat(
    [xyz, extra_features],
    dim=2
).contiguous()

# Zufällige Klassen 0-4
labels = torch.randint(
    0,
    5,
    (B, N),
    device=device
)

print("\nInput:", features.shape)
print("Labels:", labels.shape)


# ---------------------------------------------------------
# 5. Modellgewicht vor Training merken
# ---------------------------------------------------------

first_parameter = next(model.parameters())

before = first_parameter.detach().clone()


# ---------------------------------------------------------
# 6. Forward Pass
# ---------------------------------------------------------

optimizer.zero_grad()

logits = model(features)

print("Output:", logits.shape)


# ---------------------------------------------------------
# 7. Loss berechnen
# ---------------------------------------------------------

focal = focal_loss_fn(logits, labels)
dice = dice_loss_fn(logits, labels)

loss = focal + 0.2 * dice

print("\nFocal Loss:", focal.item())
print("Dice Loss:", dice.item())
print("Combined Loss:", loss.item())


# ---------------------------------------------------------
# 8. Backpropagation
# ---------------------------------------------------------

loss.backward()

print("\nBackward Pass erfolgreich")


# ---------------------------------------------------------
# 9. Modellgewichte aktualisieren
# ---------------------------------------------------------

optimizer.step()

print("Optimizer Step erfolgreich")


# ---------------------------------------------------------
# 10. Prüfen, ob sich Gewichte wirklich geändert haben
# ---------------------------------------------------------

after = first_parameter.detach()

weight_change = torch.abs(after - before).sum().item()

print("Gewichtsänderung:", weight_change)

assert weight_change > 0
assert torch.isfinite(loss)

print("\n=== TRAININGSTEST ERFOLGREICH ===")
print("Forward -> Loss -> Backward -> Optimizer funktioniert.")

PY