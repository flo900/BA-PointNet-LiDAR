import numpy as np
import geopandas as gpd
from laspy import CopcReader
from laspy.copc import Bounds

# --------------------------------------------------
# Pfade
# --------------------------------------------------

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

OUTPUT_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "conflicts_test_area.gpkg"
)

# --------------------------------------------------
# Testgebiet 100 x 100 m
# --------------------------------------------------

bounds = Bounds(
    mins=np.array([2692000, 1232700]),
    maxs=np.array([2692100, 1232800])
)

# --------------------------------------------------
# LiDAR laden
# --------------------------------------------------

with CopcReader.open(LIDAR_PATH) as reader:
    points = reader.query(bounds=bounds)

original_class = np.asarray(points.classification)

# Nur Ground (2) und Vegetation (3)
mask = np.isin(original_class, [2, 3])

lidar = gpd.GeoDataFrame(
    {
        "lidar_class": original_class[mask],
        "z": np.asarray(points.z)[mask],
        "intensity": np.asarray(points.intensity)[mask],
        "return_num": np.asarray(points.return_number)[mask],
        "num_returns": np.asarray(points.number_of_returns)[mask],
        "scan_angle": np.asarray(points.scan_angle)[mask],
    },
    geometry=gpd.points_from_xy(
        np.asarray(points.x)[mask],
        np.asarray(points.y)[mask]
    ),
    crs="EPSG:2056"
)

# Eindeutige ID pro Punkt
lidar["point_id"] = np.arange(len(lidar))

# --------------------------------------------------
# AV-Klasse zuweisen
# --------------------------------------------------

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

# --------------------------------------------------
# Konflikttyp bestimmen
# --------------------------------------------------

def conflict_type(row):

    lidar_class = row["lidar_class"]
    av_class = row["target_class"]

    if lidar_class == 3 and av_class == "Impervious":
        return "Vegetation + Impervious"

    if lidar_class == 3 and av_class == "Water":
        return "Vegetation + Water"

    if lidar_class == 3 and av_class == "Building":
        return "Vegetation + Building"

    if lidar_class == 2 and av_class == "Water":
        return "Ground + Water"

    if lidar_class == 2 and av_class == "Building":
        return "Ground + Building"

    return None


joined["conflict_type"] = joined.apply(
    conflict_type,
    axis=1
)

conflicts = joined[
    joined["conflict_type"].notna()
].copy()

# --------------------------------------------------
# Prüfen, ob Punkt innerhalb einer ZHAW-Baumkrone liegt
# --------------------------------------------------

trees = gpd.read_file(
    TREE_PATH,
    layer="Baumkronen"
)[["treeID", "geometry"]]

tree_join = gpd.sjoin(
    conflicts[["point_id", "geometry"]],
    trees,
    how="left",
    predicate="within"
)

# Für jeden Punkt: mindestens eine Baumkrone getroffen?
inside_tree = (
    tree_join
    .groupby("point_id")["treeID"]
    .apply(lambda x: x.notna().any())
)

conflicts["inside_tree_crown"] = (
    conflicts["point_id"]
    .map(inside_tree)
    .fillna(False)
)

# --------------------------------------------------
# Aufräumen und speichern
# --------------------------------------------------

conflicts = conflicts[
    [
        "point_id",
        "lidar_class",
        "target_class",
        "conflict_type",
        "inside_tree_crown",
        "z",
        "intensity",
        "return_num",
        "num_returns",
        "scan_angle",
        "geometry",
    ]
]

print("Konfliktpunkte insgesamt:", len(conflicts))
print()

print(
    conflicts["conflict_type"]
    .value_counts()
)

print("\nDavon innerhalb einer ZHAW-Baumkrone:")
print(
    conflicts.groupby("conflict_type")["inside_tree_crown"]
    .agg(["sum", "count"])
)

conflicts.to_file(
    OUTPUT_PATH,
    layer="conflicts",
    driver="GPKG"
)

print()
print("Gespeichert unter:")
print(OUTPUT_PATH)
