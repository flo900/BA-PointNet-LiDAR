import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
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
    "relabel_test_area.gpkg"
)

SUMMARY_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "relabel_test_area_summary.csv"
)


# --------------------------------------------------
# Testgebiet
# --------------------------------------------------

xmin, ymin = 2692000, 1232700
xmax, ymax = 2692100, 1232800

bounds = Bounds(
    mins=np.array([xmin, ymin]),
    maxs=np.array([xmax, ymax])
)


# --------------------------------------------------
# LiDAR laden
# --------------------------------------------------

with CopcReader.open(LIDAR_PATH) as reader:
    points = reader.query(bounds=bounds)

n = len(points)

gdf = gpd.GeoDataFrame(
    {
        "point_id": np.arange(n),
        "original_class": np.asarray(points.classification),
        "x": np.asarray(points.x),
        "y": np.asarray(points.y),
        "z": np.asarray(points.z),
        "intensity": np.asarray(points.intensity),
        "return_num": np.asarray(points.return_number),
        "num_returns": np.asarray(points.number_of_returns),
        "scan_angle": np.asarray(points.scan_angle),
    },
    geometry=gpd.points_from_xy(
        np.asarray(points.x),
        np.asarray(points.y)
    ),
    crs="EPSG:2056"
)

gdf["av_class"] = None
gdf["inside_tree_crown"] = False
gdf["final_label"] = None
gdf["status"] = "unresolved"
gdf["rule_basis"] = None


# --------------------------------------------------
# AV-Klasse nur für LiDAR 2 und 3 bestimmen
# --------------------------------------------------

mask_23 = gdf["original_class"].isin([2, 3])

target = gpd.read_file(
    TARGET_PATH,
    layer="target"
)

target = target[
    target["target_class"].notna()
][["target_class", "geometry"]]

test_area = box(xmin, ymin, xmax, ymax)

target = target[
    target.intersects(test_area)
].copy()

av_join = gpd.sjoin(
    gdf.loc[
        mask_23,
        ["point_id", "geometry"]
    ],
    target,
    how="left",
    predicate="within"
)

# Falls ein Punkt theoretisch mehrere Polygone trifft:
av_join = av_join.drop_duplicates(
    subset="point_id"
)

av_map = dict(
    zip(
        av_join["point_id"],
        av_join["target_class"]
    )
)

gdf["av_class"] = gdf["point_id"].map(av_map)


# --------------------------------------------------
# Baumkronen bestimmen
# --------------------------------------------------

trees = gpd.read_file(
    TREE_PATH,
    layer="Baumkronen"
)[["geometry"]]

trees = trees[
    trees.intersects(test_area)
].copy()

tree_join = gpd.sjoin(
    gdf[
        ["point_id", "geometry"]
    ],
    trees,
    how="left",
    predicate="within"
)

inside_tree_ids = set(
    tree_join.loc[
        tree_join["index_right"].notna(),
        "point_id"
    ]
)

gdf["inside_tree_crown"] = (
    gdf["point_id"].isin(inside_tree_ids)
)


# --------------------------------------------------
# STEP 0:
# Klassen entfernen
# --------------------------------------------------

removed = gdf["original_class"].isin(
    [1, 14, 15, 27]
)

gdf.loc[
    removed,
    "status"
] = "removed"

gdf.loc[
    removed,
    "rule_basis"
] = "documented"


# --------------------------------------------------
# STEP 1:
# Direkte LiDAR-Zuordnung
# --------------------------------------------------

mask = (
    gdf["original_class"].isin([6, 26])
    & ~removed
)

gdf.loc[mask, "final_label"] = "Buildings"
gdf.loc[mask, "status"] = "direct_lidar"
gdf.loc[mask, "rule_basis"] = "documented"


mask = (
    (gdf["original_class"] == 9)
    & ~removed
)

gdf.loc[mask, "final_label"] = "Water"
gdf.loc[mask, "status"] = "direct_lidar"
gdf.loc[mask, "rule_basis"] = "documented"


mask = (
    (gdf["original_class"] == 17)
    & ~removed
)

gdf.loc[mask, "final_label"] = "Impervious"
gdf.loc[mask, "status"] = "direct_lidar"
gdf.loc[mask, "rule_basis"] = "documented"


# --------------------------------------------------
# STEP 2:
# Dokumentierte AV-Kontextregeln
# --------------------------------------------------

# Ground + AV Tree canopy
mask = (
    (gdf["original_class"] == 2)
    & (gdf["av_class"] == "Tree canopy")
    & gdf["final_label"].isna()
)

gdf.loc[mask, "final_label"] = "Low vegetation"
gdf.loc[mask, "status"] = "av_context"
gdf.loc[mask, "rule_basis"] = "documented"


# Vegetation + AV Tree canopy
mask = (
    (gdf["original_class"] == 3)
    & (gdf["av_class"] == "Tree canopy")
    & gdf["final_label"].isna()
)

gdf.loc[mask, "final_label"] = "Tree canopy"
gdf.loc[mask, "status"] = "av_context"
gdf.loc[mask, "rule_basis"] = "documented"


# Ground + AV Impervious
mask = (
    (gdf["original_class"] == 2)
    & (gdf["av_class"] == "Impervious")
    & gdf["final_label"].isna()
)

gdf.loc[mask, "final_label"] = "Impervious"
gdf.loc[mask, "status"] = "av_context"
gdf.loc[mask, "rule_basis"] = "documented"


# --------------------------------------------------
# STEP 2b:
# Low vegetation - transparente Ergänzung
#
# Ground innerhalb einer AV-Low-Vegetation-Fläche
# wird Low vegetation.
# --------------------------------------------------

mask = (
    (gdf["original_class"] == 2)
    & (gdf["av_class"] == "Low vegetation")
    & gdf["final_label"].isna()
)

gdf.loc[mask, "final_label"] = "Low vegetation"
gdf.loc[mask, "status"] = "av_lowveg"
gdf.loc[mask, "rule_basis"] = "inferred"


# --------------------------------------------------
# STEP 3:
# Noch ungelabelte Punkte innerhalb Baumkronen
# --------------------------------------------------

# Vegetation innerhalb Baumkrone
mask = (
    (gdf["original_class"] == 3)
    & (gdf["inside_tree_crown"])
    & gdf["final_label"].isna()
)

gdf.loc[mask, "final_label"] = "Tree canopy"
gdf.loc[mask, "status"] = "tree_refinement"
gdf.loc[mask, "rule_basis"] = "documented"


# Ground innerhalb Baumkrone
mask = (
    (gdf["original_class"] == 2)
    & (gdf["inside_tree_crown"])
    & gdf["final_label"].isna()
)

gdf.loc[mask, "final_label"] = "Low vegetation"
gdf.loc[mask, "status"] = "tree_refinement"
gdf.loc[mask, "rule_basis"] = "documented"


# --------------------------------------------------
# STEP 4:
# Vegetation + AV Low vegetation ausserhalb Baumkrone
#
# Wenn keine Baumkrone vorhanden ist:
# Low vegetation.
# --------------------------------------------------

mask = (
    (gdf["original_class"] == 3)
    & (gdf["av_class"] == "Low vegetation")
    & (~gdf["inside_tree_crown"])
    & gdf["final_label"].isna()
)

gdf.loc[mask, "final_label"] = "Low vegetation"
gdf.loc[mask, "status"] = "av_lowveg"
gdf.loc[mask, "rule_basis"] = "inferred"


# --------------------------------------------------
# Verbleibende ungelabelte Punkte
# --------------------------------------------------

unresolved = (
    gdf["final_label"].isna()
    & ~removed
)

gdf.loc[
    unresolved,
    "status"
] = "unresolved_conflict"

gdf.loc[
    unresolved,
    "rule_basis"
] = "not_assigned"


# --------------------------------------------------
# Statistik
# --------------------------------------------------

print("Punkte insgesamt:", len(gdf))
print()

print("Finale Klassen:")
print(
    gdf["final_label"]
    .value_counts(dropna=False)
)

print("\nStatus:")
print(
    gdf["status"]
    .value_counts()
)

print("\nRegelgrundlage:")
print(
    gdf["rule_basis"]
    .value_counts(dropna=False)
)

print("\nUngelöste Konflikte:")
print(
    gdf.loc[
        gdf["status"] == "unresolved_conflict",
        ["original_class", "av_class"]
    ]
    .value_counts()
)


# --------------------------------------------------
# Speichern
# --------------------------------------------------

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

gdf.to_file(
    OUTPUT_PATH,
    layer="relabel_test_area",
    driver="GPKG"
)

summary = (
    gdf["final_label"]
    .value_counts(dropna=False)
    .rename_axis("final_label")
    .reset_index(name="count")
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

print("\nGeoPackage gespeichert:")
print(OUTPUT_PATH)

print("\nZusammenfassung gespeichert:")
print(SUMMARY_PATH)
