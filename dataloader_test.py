from src.PointnetPP.lidar_dataloader import (
    get_dataloaders,
    Compose,
    RandomRotation,
    RandomJitter,
    RandomScale
)

DATA_DIR = "/cfs/earth/scratch/troxlflo/BA/dummy_lidar"

print("=== Dataloader erstellen ===")

augmentation = Compose([
    RandomRotation(max_angle=180),
    RandomJitter(sigma=0.01, clip=0.05),
    RandomScale(scale_low=0.8, scale_high=1.2)
])

train_loader, val_loader, test_loader, metadata = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=4,
    num_workers=0,
    pin_memory=False,
    train_transform=augmentation
)

print("\n=== Einen Trainings-Batch laden ===")

features, labels = next(iter(train_loader))

print("Features:", features.shape)
print("Features dtype:", features.dtype)

print("Labels:", labels.shape)
print("Labels dtype:", labels.dtype)

print("Min Label:", labels.min().item())
print("Max Label:", labels.max().item())

print("\n=== Metadata ===")
print(metadata)

print("\n=== DATALOADER TEST ERFOLGREICH ===")