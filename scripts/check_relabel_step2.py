import numpy as np
import geopandas as gpd
from laspy import CopcReader
from laspy.copc import Bounds

LIDAR_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/raw/lidar/"
    "swisssurface3d_2024_2692-1232_2056_5728.copc.laz"
)

TARGET_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/target.gpkg"
)

bounds = Bounds(
    mins=np.array([2692000, 1232700]),
    maxs=np.array([2692100, 1232800])
)

# LiDAR laden
with CopcReader.open(LIDAR_PATH) as reader:
    points = reader.query(bounds=bounds)

classes = np.asarray(points.classification)

# Nur Klassen 2 und 3:
mask = np.isin(classes, [2, 3])

x = np.asarray(points.x)[mask]
y = np.asarray(points.y)[mask]
classes = classes[mask]

# Punkt-Geodataframe erstellen
lidar = gpd.GeoDataFrame(
    {
        "lidar_class": classes
    },
    geometry=gpd.points_from_xy(x, y),
    crs="EPSG:2056"
)

# target.gpkg laden
target = gpd.read_file(
    TARGET_PATH,
    layer="target"
)

target = target[
    target["target_class"].notna()
][["target_class", "geometry"]]

# Räumliche Zuordnung
joined = gpd.sjoin(
    lidar,
    target,
    how="left",
    predicate="within"
)

# Kombinationen zählen
table = (
    joined
    .groupby(
        ["lidar_class", "target_class"],
        dropna=False
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        ["lidar_class", "count"],
        ascending=[True, False]
    )
)

print(table.to_string(index=False))
