import json
import os
import numpy as np


# Speicherort unserer künstlichen Testdaten
DATA_DIR = "/cfs/earth/scratch/troxlflo/BA/dummy_lidar"

# Nicolas Datenformat
POINTS_PER_TILE = 16384
NUM_FEATURES = 7
NUM_CLASSES = 5

# Nur wenige Tiles, weil es lediglich ein Test ist
SPLITS = {
    "train": 8,
    "val": 2,
    "test": 2
}

os.makedirs(DATA_DIR, exist_ok=True)


def create_split(split, num_tiles):
    print(f"Erstelle {split}: {num_tiles} Tiles")

    # Form:
    # (Anzahl Tiles, 16384 Punkte, 7 Features)
    features = np.random.rand(
        num_tiles,
        POINTS_PER_TILE,
        NUM_FEATURES
    ).astype(np.float32)

    # X und Y ungefähr auf Nicolas 25 x 25 m Tiles skalieren
    features[:, :, 0] *= 25.0
    features[:, :, 1] *= 25.0

    # Z für den Test ungefähr 0-20 m
    features[:, :, 2] *= 20.0

    # Zufällige Klassen 0-4
    labels = np.random.randint(
        0,
        NUM_CLASSES,
        size=(num_tiles, POINTS_PER_TILE),
        dtype=np.int64
    )

    np.save(
        os.path.join(DATA_DIR, f"{split}_features.npy"),
        features
    )

    np.save(
        os.path.join(DATA_DIR, f"{split}_labels.npy"),
        labels
    )


for split, num_tiles in SPLITS.items():
    create_split(split, num_tiles)


metadata = {
    "points_per_tile": POINTS_PER_TILE,
    "num_features": NUM_FEATURES,
    "num_classes": NUM_CLASSES,
    "classes": {
        "0": "Water",
        "1": "Tree canopy",
        "2": "Low vegetation",
        "3": "Impervious",
        "4": "Buildings"
    }
}

with open(
    os.path.join(DATA_DIR, "metadata.json"),
    "w"
) as f:
    json.dump(metadata, f, indent=4)


print("\nTestdatensatz erstellt:")
print(DATA_DIR)

for filename in sorted(os.listdir(DATA_DIR)):
    print(" -", filename)