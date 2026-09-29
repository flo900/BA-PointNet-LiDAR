import numpy as np
import pandas as pd
import geopandas as gpd

CONFLICT_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "conflicts_test_area.gpkg"
)

TARGET_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "target.gpkg"
)

OUTPUT_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "conflicts_distance_test_area.gpkg"
)

# --------------------------------------------------
# Konflikte laden
# --------------------------------------------------

conflicts = gpd.read_file(
    CONFLICT_PATH,
    layer="conflicts"
)

# Nur Konflikte ausserhalb bekannter Baumkronen
conflicts = conflicts[
    conflicts["inside_tree_crown"] == False
].copy()

print("Untersuchte Konfliktpunkte:", len(conflicts))


# --------------------------------------------------
# AV-Polygone laden
# --------------------------------------------------

target = gpd.read_file(
    TARGET_PATH,
    layer="target"
)

target = target[
    target["target_class"].notna()
][["target_class", "geometry"]].copy()

# Eigene Polygon-ID
target["av_poly_id"] = target.index


# --------------------------------------------------
# Jedem Punkt sein AV-Polygon zuweisen
# --------------------------------------------------

target_join = target.rename(
    columns={"target_class": "av_join_class"}
)

joined = gpd.sjoin(
    conflicts,
    target_join,
    how="left",
    predicate="within"
)

# --------------------------------------------------
# Distanz zur Grenze des jeweiligen AV-Polygons
# --------------------------------------------------

distances = []

for point, polygon_index in zip(
    joined.geometry,
    joined["index_right"]
):
    if pd.isna(polygon_index):
        distances.append(np.nan)
    else:
        polygon = target.loc[
            int(polygon_index),
            "geometry"
        ]

        distance = point.distance(
            polygon.boundary
        )

        distances.append(distance)

joined["dist_to_av_edge_m"] = distances


# --------------------------------------------------
# Distanzklassen
# --------------------------------------------------

joined["edge_band"] = pd.cut(
    joined["dist_to_av_edge_m"],
    bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
    labels=[
        "0-0.5 m",
        "0.5-1 m",
        "1-2 m",
        ">2 m",
    ]
)


# --------------------------------------------------
# Ergebnis ausgeben
# --------------------------------------------------

print("\nDistanz zur AV-Grenze insgesamt:")
print(
    joined["edge_band"]
    .value_counts()
    .sort_index()
)

print("\nNach Konflikttyp:")
print(
    pd.crosstab(
        joined["conflict_type"],
        joined["edge_band"]
    )
)

# --------------------------------------------------
# Speichern
# --------------------------------------------------

joined = joined.drop(
    columns=["index_right"],
    errors="ignore"
)

joined.to_file(
    OUTPUT_PATH,
    layer="conflicts_distance",
    driver="GPKG"
)

print("\nGespeichert unter:")
print(OUTPUT_PATH)
