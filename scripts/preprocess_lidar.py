from pathlib import Path
import argparse
import re
import yaml
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from laspy import CopcReader
from laspy.copc import Bounds
from scipy.spatial import cKDTree


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    required=True,
    help="Pfad zur YAML-Konfiguration"
)

parser.add_argument(
    "--tile-index",
    type=int,
    required=True,
    help="Index der 1-km²-COPC-Kachel"
)

parser.add_argument(
    "--chunk-index",
    type=int,
    default=None,
    help="Optional: nur einen 100-m-Chunk testen"
)

args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

lidar_files = cfg["lidar"]["files"]
lidar_dir = Path(cfg["lidar"]["directory"])
target_path = Path(cfg["av"]["path"])
target_layer = cfg["av"]["layer"]

tree_path = Path(cfg["tree_canopy"]["path"])
tree_layer = cfg["tree_canopy"]["layer"]
GROUND_BUFFER = cfg["ground_height"]["buffer_m"]
GROUND_K = cfg["ground_height"]["k_neighbors"]

if not 0 <= args.tile_index < len(lidar_files):
    raise ValueError(
        f"tile-index muss zwischen 0 und {len(lidar_files) - 1} liegen."
    )

filename = lidar_files[args.tile_index]
lidar_path = lidar_dir / filename

if not lidar_path.exists():
    raise FileNotFoundError(f"COPC-Datei nicht gefunden: {lidar_path}")

match = re.search(r"_(\d+)-(\d+)_2056_", filename)

if match is None:
    raise ValueError(
        f"Kachelkoordinaten konnten nicht aus {filename} gelesen werden."
    )

tile_x = int(match.group(1)) * 1000
tile_y = int(match.group(2)) * 1000

print("Projekt:", cfg["project"]["name"])
print("Tile-Index:", args.tile_index)
print("COPC:", lidar_path)

print()
print("1-km²-Kachel:")
print("  xmin:", tile_x)
print("  xmax:", tile_x + 1000)
print("  ymin:", tile_y)
print("  ymax:", tile_y + 1000)

CHUNK_SIZE = 100
chunks = []
chunk_id = 0

for y0 in range(tile_y, tile_y + 1000, CHUNK_SIZE):
    for x0 in range(tile_x, tile_x + 1000, CHUNK_SIZE):
        chunks.append({
            "chunk_id": chunk_id,
            "xmin": x0,
            "xmax": x0 + CHUNK_SIZE,
            "ymin": y0,
            "ymax": y0 + CHUNK_SIZE,
        })
        chunk_id += 1

print()
print("Interne Chunks:", len(chunks))
print("Chunk-Grösse:", CHUNK_SIZE, "m")

print()
print("Erster Chunk:")
print(chunks[0])

print()
print("Letzter Chunk:")
print(chunks[-1])

if args.chunk_index is not None:

    if not 0 <= args.chunk_index < len(chunks):
        raise ValueError(
            f"chunk-index muss zwischen 0 und {len(chunks) - 1} liegen."
        )

    chunk = chunks[args.chunk_index]

    # Zielpunkte aus der gewählten 1-km²-COPC-Kachel laden
    bounds = Bounds(
        mins=np.array([chunk["xmin"], chunk["ymin"]]),
        maxs=np.array([chunk["xmax"], chunk["ymax"]])
    )

    with CopcReader.open(lidar_path) as reader:
        points = reader.query(bounds=bounds)

    x = np.asarray(points.x)
    y = np.asarray(points.y)
    z = np.asarray(points.z)
    classes = np.asarray(points.classification)

    # half-open: links/unten inklusive, rechts/oben exklusive
    target_mask = (
        (x >= chunk["xmin"]) &
        (x < chunk["xmax"]) &
        (y >= chunk["ymin"]) &
        (y < chunk["ymax"])
    )

    x = x[target_mask]
    y = y[target_mask]
    z = z[target_mask]
    classes = classes[target_mask]
    # --------------------------------------------------
    # GeoDataFrame für räumliche Zuordnung
    # --------------------------------------------------

    n = len(x)

    gdf = gpd.GeoDataFrame(
        {
            "point_id": np.arange(n),
            "original_class": classes,
            "x": x,
            "y": y,
            "z": z,
        },
        geometry=gpd.points_from_xy(
            x,
            y
        ),
        crs=cfg["project"]["crs"]
    )

    gdf["av_class"] = None
    gdf["inside_tree_crown"] = False

    chunk_geom = box(
        chunk["xmin"],
        chunk["ymin"],
        chunk["xmax"],
        chunk["ymax"]
    )


    # --------------------------------------------------
    # AV-Klasse für LiDAR-Klassen 2 und 3 bestimmen
    # --------------------------------------------------

    mask_23 = gdf["original_class"].isin([2, 3])

    target = gpd.read_file(
        target_path,
        layer=target_layer
    )

    target = target[
        target["target_class"].notna()
    ][["target_class", "geometry"]]

    target = target[
        target.intersects(chunk_geom)
    ].copy()

    if not target.empty and mask_23.any():

        av_join = gpd.sjoin(
            gdf.loc[
                mask_23,
                ["point_id", "geometry"]
            ],
            target,
            how="left",
            predicate="within"
        )

        av_join = av_join.drop_duplicates(
            subset="point_id"
        )

        av_map = dict(
            zip(
                av_join["point_id"],
                av_join["target_class"]
            )
        )

        gdf["av_class"] = (
            gdf["point_id"].map(av_map)
        )


    # --------------------------------------------------
    # Baumkronen bestimmen
    # --------------------------------------------------

    trees = gpd.read_file(
        tree_path,
        layer=tree_layer
    )[["geometry"]]

    trees = trees[
        trees.intersects(chunk_geom)
    ].copy()

    if not trees.empty:

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
            gdf["point_id"].isin(
                inside_tree_ids
            )
        )


    # --------------------------------------------------
    # QA
    # --------------------------------------------------

    print()
    print("AV-ZUORDNUNG:")
    print(
        gdf["av_class"]
        .value_counts(dropna=False)
    )

    print()
    print(
        "Punkte innerhalb Baumkronen:",
        int(gdf["inside_tree_crown"].sum())
    )
    unique, counts = np.unique(classes, return_counts=True)

    print()
    print("CHUNK-TEST")
    print("Chunk:", args.chunk_index)
    print("Punkte:", len(classes))
    print("Originalklassen:")

    for cls, count in zip(unique, counts):
        print(f"  Klasse {cls}: {count}")

    # COPC-Dateien für Ground-Puffer bestimmen
    ground_xmin = chunk["xmin"] - GROUND_BUFFER
    ground_xmax = chunk["xmax"] + GROUND_BUFFER
    ground_ymin = chunk["ymin"] - GROUND_BUFFER
    ground_ymax = chunk["ymax"] + GROUND_BUFFER

    ground_copc_files = []

    for path in lidar_dir.glob("*.copc.laz"):
        source_match = re.search(r"_(\d+)-(\d+)_2056_", path.name)

        if source_match is None:
            continue

        source_xmin = int(source_match.group(1)) * 1000
        source_ymin = int(source_match.group(2)) * 1000
        source_xmax = source_xmin + 1000
        source_ymax = source_ymin + 1000

        overlaps = not (
            source_xmax <= ground_xmin or
            source_xmin >= ground_xmax or
            source_ymax <= ground_ymin or
            source_ymin >= ground_ymax
        )

        if overlaps:
            ground_copc_files.append(path)

    ground_copc_files = sorted(ground_copc_files)

    if not ground_copc_files:
        raise RuntimeError("Keine COPC-Datei für den Ground-Puffer gefunden.")

    print()
    print("Ground-Puffer:")
    print(ground_xmin, ground_xmax, ground_ymin, ground_ymax)

    print()
    print("Benötigte COPC-Dateien für Ground:")
    for path in ground_copc_files:
        print(" ", path.name)

    # Ground-Punkte aus allen benötigten COPCs laden
    ground_bounds = Bounds(
        mins=np.array([ground_xmin, ground_ymin]),
        maxs=np.array([ground_xmax, ground_ymax])
    )

    ground_x_parts = []
    ground_y_parts = []
    ground_z_parts = []

    for ground_path in ground_copc_files:
        with CopcReader.open(ground_path) as reader:
            ground_points = reader.query(bounds=ground_bounds)

        gx = np.asarray(ground_points.x)
        gy = np.asarray(ground_points.y)
        gz = np.asarray(ground_points.z)
        gclass = np.asarray(ground_points.classification)

        ground_mask = (
            (gclass == 2) &
            (gx >= ground_xmin) &
            (gx < ground_xmax) &
            (gy >= ground_ymin) &
            (gy < ground_ymax)
        )

        if np.any(ground_mask):
            ground_x_parts.append(gx[ground_mask])
            ground_y_parts.append(gy[ground_mask])
            ground_z_parts.append(gz[ground_mask])

    if not ground_x_parts:
        raise RuntimeError(
            "Keine Ground-Punkte (Klasse 2) im gepufferten Gebiet gefunden."
        )

    ground_x = np.concatenate(ground_x_parts)
    ground_y = np.concatenate(ground_y_parts)
    ground_z = np.concatenate(ground_z_parts)

    print()
    print("Ground-Punkte im 25-m-Puffer:", len(ground_x))

    if len(ground_x) < GROUND_K:
        raise RuntimeError(
            f"Nur {len(ground_x)} Ground-Punkte gefunden, aber k={GROUND_K} benötigt."
        )

    # Height AGL berechnen
    ground_xy = np.column_stack([ground_x, ground_y])
    ground_tree = cKDTree(ground_xy)

    point_xy = np.column_stack([x, y])

    distances, indices = ground_tree.query(
        point_xy,
        k=GROUND_K
    )

    neighbor_z = ground_z[indices]

    safe_distances = np.maximum(distances, 0.01)
    weights = 1.0 / safe_distances

    estimated_ground_z = (
        np.sum(neighbor_z * weights, axis=1)
        /
        np.sum(weights, axis=1)
    )

    height_agl = z - estimated_ground_z

    print()
    print("HEIGHT-AGL-TEST")
    print("Nearest Ground Distance:")
    print(
        "  Median:",
        round(float(np.median(distances[:, 0])), 3),
        "m"
    )
    print(
        "  P95:",
        round(float(np.percentile(distances[:, 0], 95)), 3),
        "m"
    )

    print()
    print("Height AGL:")
    print(
        "  Min:",
        round(float(np.min(height_agl)), 3),
        "m"
    )
    print(
        "  Median:",
        round(float(np.median(height_agl)), 3),
        "m"
    )
    print(
        "  Max:",
        round(float(np.max(height_agl)), 3),
        "m"
    )
    # --------------------------------------------------
    # Relabeling vorbereiten
    # --------------------------------------------------

    gdf["final_label"] = None
    gdf["status"] = "unresolved"
    gdf["rule_basis"] = None


    # --------------------------------------------------
    # STEP 0:
    # Nicht verwendete LiDAR-Klassen entfernen
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

    # Building / Building facade
    mask = (
        gdf["original_class"].isin([6, 26])
        & ~removed
    )

    gdf.loc[mask, "final_label"] = "Buildings"
    gdf.loc[mask, "status"] = "direct_lidar"
    gdf.loc[mask, "rule_basis"] = "documented"


    # Water
    mask = (
        (gdf["original_class"] == 9)
        & ~removed
    )

    gdf.loc[mask, "final_label"] = "Water"
    gdf.loc[mask, "status"] = "direct_lidar"
    gdf.loc[mask, "rule_basis"] = "documented"


    # Bridge deck -> Impervious
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
    # Eigene transparente Ergänzung für Low vegetation
    # --------------------------------------------------

    # Ground + AV Low vegetation
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
        & gdf["inside_tree_crown"]
        & gdf["final_label"].isna()
    )

    gdf.loc[mask, "final_label"] = "Tree canopy"
    gdf.loc[mask, "status"] = "tree_refinement"
    gdf.loc[mask, "rule_basis"] = "documented"


    # Ground innerhalb Baumkrone
    mask = (
        (gdf["original_class"] == 2)
        & gdf["inside_tree_crown"]
        & gdf["final_label"].isna()
    )

    gdf.loc[mask, "final_label"] = "Low vegetation"
    gdf.loc[mask, "status"] = "tree_refinement"
    gdf.loc[mask, "rule_basis"] = "documented"


    # --------------------------------------------------
    # STEP 4:
    # Vegetation + AV Low vegetation ausserhalb Baumkrone
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
    # Relabeling-QA
    # --------------------------------------------------

    print()
    print("RELABEL-TEST")

    print()
    print("Finale Klassen:")
    print(
        gdf["final_label"]
        .value_counts(dropna=False)
    )

    print()
    print("Status:")
    print(
        gdf["status"]
        .value_counts()
    )

    print()
    print("Regelgrundlage:")
    print(
        gdf["rule_basis"]
        .value_counts(dropna=False)
    )

    print()
    print("Ungelöste Konflikte:")
    print(
        gdf.loc[
            gdf["status"] == "unresolved_conflict",
            ["original_class", "av_class"]
        ]
        .value_counts()
    )
