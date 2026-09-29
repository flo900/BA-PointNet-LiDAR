import os
import numpy as np
import geopandas as gpd

from scipy.spatial import cKDTree
from laspy import CopcReader
from laspy.copc import Bounds


# --------------------------------------------------
# Pfade
# --------------------------------------------------

LIDAR_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/raw/lidar/"
    "swisssurface3d_2024_2692-1232_2056_5728.copc.laz"
)

INPUT_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "relabel_test_area.gpkg"
)

OUTPUT_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "ground_height_test_area.gpkg"
)


# --------------------------------------------------
# Testgebiet + Puffer
# --------------------------------------------------

xmin, ymin = 2692000, 1232700
xmax, ymax = 2692100, 1232800

BUFFER = 25


# --------------------------------------------------
# Punkte unseres Testgebiets laden
# --------------------------------------------------

gdf = gpd.read_file(
    INPUT_PATH,
    layer="relabel_test_area"
)

# Nur echtes 100x100-m-Gebiet
gdf = gdf[
    (gdf["x"] >= xmin) &
    (gdf["x"] < xmax) &
    (gdf["y"] >= ymin) &
    (gdf["y"] < ymax)
].copy()

print("Punkte im Testgebiet:", len(gdf))


# --------------------------------------------------
# LiDAR mit 25-m-Puffer laden
# --------------------------------------------------

bounds = Bounds(
    mins=np.array([
        xmin - BUFFER,
        ymin - BUFFER
    ]),
    maxs=np.array([
        xmax + BUFFER,
        ymax + BUFFER
    ])
)

with CopcReader.open(LIDAR_PATH) as reader:
    buffered_points = reader.query(
        bounds=bounds
    )

classes = np.asarray(
    buffered_points.classification
)

ground_mask = classes == 2

ground_x = np.asarray(
    buffered_points.x
)[ground_mask]

ground_y = np.asarray(
    buffered_points.y
)[ground_mask]

ground_z = np.asarray(
    buffered_points.z
)[ground_mask]

print(
    "Ground-Punkte im gepufferten Gebiet:",
    len(ground_x)
)


# --------------------------------------------------
# KDTree aus Ground-Punkten bauen
# --------------------------------------------------

ground_xy = np.column_stack([
    ground_x,
    ground_y
])

tree = cKDTree(
    ground_xy
)


# --------------------------------------------------
# Für jeden Punkt 8 nächste Ground-Punkte suchen
# --------------------------------------------------

point_xy = np.column_stack([
    gdf["x"].to_numpy(),
    gdf["y"].to_numpy()
])

distances, indices = tree.query(
    point_xy,
    k=8
)


# --------------------------------------------------
# Bodenhöhe distanzgewichtet schätzen
# --------------------------------------------------

neighbor_z = ground_z[
    indices
]

# Verhindert Division durch 0,
# wenn ein Punkt selbst ein Ground-Punkt ist
safe_distances = np.maximum(
    distances,
    0.01
)

weights = (
    1.0 / safe_distances
)

estimated_ground_z = (
    np.sum(
        neighbor_z * weights,
        axis=1
    )
    /
    np.sum(
        weights,
        axis=1
    )
)


# --------------------------------------------------
# Höhe über lokalem Boden
# --------------------------------------------------

gdf["ground_z_est"] = (
    estimated_ground_z
)

gdf["ground_distance_m"] = (
    distances[:, 0]
)

gdf["height_agl_m"] = (
    gdf["z"] -
    gdf["ground_z_est"]
)


# --------------------------------------------------
# Qualitätskontrolle
# --------------------------------------------------

print()
print("Distanz zum nächsten Ground-Punkt:")

for p in [50, 90, 95, 99, 100]:
    print(
        f"{p}%:",
        round(
            np.percentile(
                gdf["ground_distance_m"],
                p
            ),
            2
        ),
        "m"
    )

print()
print("Höhe über Boden:")

print(
    gdf["height_agl_m"]
    .describe()
)


# --------------------------------------------------
# Speichern
# --------------------------------------------------

if os.path.exists(
    OUTPUT_PATH
):
    os.remove(
        OUTPUT_PATH
    )

gdf.to_file(
    OUTPUT_PATH,
    layer="ground_height_test_area",
    driver="GPKG"
)

print()
print("Gespeichert:")
print(OUTPUT_PATH)
