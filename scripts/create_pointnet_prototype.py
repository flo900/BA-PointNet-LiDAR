import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd


# --------------------------------------------------
# Einstellungen
# --------------------------------------------------

INPUT_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "ground_height_test_area.gpkg"
)

OUTPUT_DIR = (
    "/cfs/earth/scratch/troxlflo/BA/data/processed/"
    "pointnet_prototype"
)

XMIN = 2692000
YMIN = 1232700

TILE_SIZE = 25
POINTS_PER_TILE = 16384

RANDOM_SEED = 42


# --------------------------------------------------
# Klassen
# --------------------------------------------------

CLASS_MAP = {
    "Water": 0,
    "Tree canopy": 1,
    "Low vegetation": 2,
    "Impervious": 3,
    "Buildings": 4,
}


# --------------------------------------------------
# Daten laden
# --------------------------------------------------

gdf = gpd.read_file(
    INPUT_PATH,
    layer="ground_height_test_area"
)

print("Punkte geladen:", len(gdf))


# Nur Punkte mit finalem Label
usable = gdf[
    gdf["final_label"].notna()
].copy()

print("Verwendbare Punkte:", len(usable))


# --------------------------------------------------
# 25 x 25 m Tiles bestimmen
# --------------------------------------------------

usable["tile_x"] = np.floor(
    (usable["x"] - XMIN) / TILE_SIZE
).astype(int)

usable["tile_y"] = np.floor(
    (usable["y"] - YMIN) / TILE_SIZE
).astype(int)

usable["tile_id"] = (
    usable["tile_x"].astype(str)
    + "_"
    + usable["tile_y"].astype(str)
)


# --------------------------------------------------
# Nur Tiles mit mindestens 16'384 Punkten
# --------------------------------------------------

tile_counts = (
    usable.groupby("tile_id")
    .size()
)

valid_tile_ids = (
    tile_counts[
        tile_counts >= POINTS_PER_TILE
    ]
    .index
    .tolist()
)

valid_tile_ids = sorted(
    valid_tile_ids
)

print()
print("Gültige Tiles:", len(valid_tile_ids))
print(valid_tile_ids)


# --------------------------------------------------
# Arrays vorbereiten
# --------------------------------------------------

all_features = []
all_labels = []
tile_metadata = []

rng = np.random.default_rng(
    RANDOM_SEED
)


# --------------------------------------------------
# Jedes gültige Tile bearbeiten
# --------------------------------------------------

for tile_id in valid_tile_ids:

    tile = usable[
        usable["tile_id"] == tile_id
    ].copy()

    n_points = len(tile)

    # Zufällig exakt 16'384 Punkte auswählen
    selected_positions = rng.choice(
        n_points,
        size=POINTS_PER_TILE,
        replace=False
    )

    tile_sample = tile.iloc[
        selected_positions
    ].copy()

    tile_x = int(
        tile_sample["tile_x"].iloc[0]
    )

    tile_y = int(
        tile_sample["tile_y"].iloc[0]
    )

    # Untere linke Ecke des Tiles
    tile_xmin = (
        XMIN
        + tile_x * TILE_SIZE
    )

    tile_ymin = (
        YMIN
        + tile_y * TILE_SIZE
    )


    # ----------------------------------------------
    # Lokale XYZ-Koordinaten
    # ----------------------------------------------

    x_local = (
        tile_sample["x"].to_numpy()
        - tile_xmin
    )

    y_local = (
        tile_sample["y"].to_numpy()
        - tile_ymin
    )

    z_local = (
        tile_sample["height_agl_m"]
        .to_numpy()
    )


    # ----------------------------------------------
    # 7 Features
    # ----------------------------------------------

    features = np.column_stack([
        x_local,
        y_local,
        z_local,
        tile_sample["intensity"].to_numpy(),
        tile_sample["return_num"].to_numpy(),
        tile_sample["num_returns"].to_numpy(),
        tile_sample["scan_angle"].to_numpy(),
    ]).astype(np.float32)


    # ----------------------------------------------
    # Labels
    # ----------------------------------------------

    labels = (
        tile_sample["final_label"]
        .map(CLASS_MAP)
        .to_numpy(dtype=np.int64)
    )


    all_features.append(
        features
    )

    all_labels.append(
        labels
    )


    # ----------------------------------------------
    # Tile-Metadaten
    # ----------------------------------------------

    tile_metadata.append({
        "tile_id": tile_id,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "tile_xmin": tile_xmin,
        "tile_ymin": tile_ymin,
        "points_before_sampling": n_points,
        "points_after_sampling": POINTS_PER_TILE,
    })


# --------------------------------------------------
# Zu NumPy-Arrays zusammenfügen
# --------------------------------------------------

features_array = np.stack(
    all_features,
    axis=0
)

labels_array = np.stack(
    all_labels,
    axis=0
)


print()
print("Features shape:")
print(features_array.shape)

print()
print("Labels shape:")
print(labels_array.shape)

print()
print("Features dtype:")
print(features_array.dtype)

print()
print("Labels dtype:")
print(labels_array.dtype)


# --------------------------------------------------
# Ausgabeordner erstellen
# --------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Speichern
# --------------------------------------------------

np.save(
    os.path.join(
        OUTPUT_DIR,
        "prototype_features.npy"
    ),
    features_array
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "prototype_labels.npy"
    ),
    labels_array
)


# Tile-Herkunft speichern
pd.DataFrame(
    tile_metadata
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "prototype_tiles.csv"
    ),
    index=False
)


# --------------------------------------------------
# Metadata
# --------------------------------------------------

metadata = {
    "points_per_tile": POINTS_PER_TILE,
    "num_features": 7,
    "num_classes": 5,
    "tile_size_m": TILE_SIZE,
    "random_seed": RANDOM_SEED,

    "features": [
        "X_local_m",
        "Y_local_m",
        "Height_AGL_m",
        "Intensity",
        "ReturnNumber",
        "NumberOfReturns",
        "ScanAngle",
    ],

    "classes": {
        "0": "Water",
        "1": "Tree canopy",
        "2": "Low vegetation",
        "3": "Impervious",
        "4": "Buildings",
    },

    "z_method": (
        "Height above locally estimated ground; "
        "8 nearest original SwissSURFACE3D Ground "
        "points, inverse-distance weighted."
    )
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    ),
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print()
print("Gespeichert in:")
print(OUTPUT_DIR)
