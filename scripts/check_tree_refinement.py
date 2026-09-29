import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from laspy import CopcReader
from laspy.copc import Bounds

LIDAR_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/raw/lidar/"
    "swisssurface3d_2024_2692-1232_2056_5728.copc.laz"
)

TARGET_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/target.gpkg"
)

TREE_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/raw/tree_canopy/Waedi_Veg.gdb"
)

# 100 x 100 m Testgebiet
xmin, ymin = 2692000, 1232700
xmax, ymax = 2692100, 1232800

bounds = Bounds(
    mins=np.array([xmin, ymin]),
    maxs=np.array([xmax, ymax])
)

# LiDAR laden
with CopcReader.open(LIDAR_PATH) as reader:
    points = reader.query(bounds=bounds)

classes = np.asarray(points.classification)

# Nur Ground (2) und Vegetation (3)
mask = np.isin(classes, [2, 3])

lidar = gpd.GeoDataFrame(
    {
        "point_id": np.arange(mask.sum()),
        "lidar_class": classes[mask],
    },
    geometry=gpd.points_from_xy(
        np.asarray(points.x)[mask],
        np.asarray(points.y)[mask]
    ),
    crs="EPSG:2056"
)

# AV laden
target = gpd.read_file(
    TARGET_PATH,
    layer="target"
)

target = target[
    target["target_class"].notna()
][["target_class", "geometry"]]

joined = gpd.sjoin(
    lidar,
    target,
    how="left",
    predicate="within"
)

# Baumkronen laden und auf Testgebiet beschränken
trees = gpd.read_file(
    TREE_PATH,
    layer="Baumkronen"
)[["geometry"]]

test_area = box(xmin, ymin, xmax, ymax)

trees = trees[
    trees.intersects(test_area)
].copy()

# Prüfen, welche Punkte innerhalb einer Baumkrone liegen
tree_join = gpd.sjoin(
    lidar[["point_id", "geometry"]],
    trees,
    how="left",
    predicate="within"
)

inside_ids = set(
    tree_join.loc[
        tree_join["index_right"].notna(),
        "point_id"
    ]
)

joined["inside_tree_crown"] = (
    joined["point_id"].isin(inside_ids)
)

# Übersicht
table = pd.crosstab(
    [
        joined["lidar_class"],
        joined["target_class"]
    ],
    joined["inside_tree_crown"]
)

table.columns = [
    "outside_tree_crown" if x is False else "inside_tree_crown"
    for x in table.columns
]

table["total"] = table.sum(axis=1)

print(table)
