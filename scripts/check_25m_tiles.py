import geopandas as gpd
import pandas as pd
import numpy as np

INPUT_PATH = (
    "/cfs/earth/scratch/troxlflo/BA/data/intermediate/"
    "relabel_test_area.gpkg"
)

TILE_SIZE = 25

XMIN = 2692000
YMIN = 1232700

# --------------------------------------------------
# Daten laden
# --------------------------------------------------

gdf = gpd.read_file(
    INPUT_PATH,
    layer="relabel_test_area"
)

print("Punkte insgesamt:", len(gdf))

# Nur Punkte verwenden, die ein finales Label besitzen
usable = gdf[
    gdf["final_label"].notna()
].copy()

# Nur Punkte innerhalb des echten 100x100-m-Testgebiets
usable = usable[
    (usable["x"] >= XMIN) &
    (usable["x"] < XMIN + 100) &
    (usable["y"] >= YMIN) &
    (usable["y"] < YMIN + 100)
].copy()

print("Verwendbare Punkte:", len(usable))
print()


# --------------------------------------------------
# 25 x 25 m Tile bestimmen
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
# Punktzahl pro Tile
# --------------------------------------------------

counts = (
    usable.groupby("tile_id")
    .size()
    .reset_index(name="points")
    .sort_values("tile_id")
)

counts["enough_for_16384"] = (
    counts["points"] >= 16384
)

print("Punkte pro 25x25-m-Tile:")
print()
print(counts.to_string(index=False))

print()
print(
    "Tiles mit >= 16'384 Punkten:",
    counts["enough_for_16384"].sum(),
    "/",
    len(counts)
)

print(
    "Minimum:",
    counts["points"].min()
)

print(
    "Maximum:",
    counts["points"].max()
)

print(
    "Mittelwert:",
    round(counts["points"].mean(), 1)
)


# --------------------------------------------------
# Klassen pro Tile
# --------------------------------------------------

class_table = pd.crosstab(
    usable["tile_id"],
    usable["final_label"]
)

print()
print("Klassenverteilung pro Tile:")
print()
print(class_table)
