from pathlib import Path
import argparse
import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from laspy import CopcReader
from laspy.copc import Bounds
from scipy.spatial import cKDTree
from shapely.geometry import box


CHUNK_SIZE = 100

LABEL_MAP = {
    "Water": 0,
    "Tree canopy": 1,
    "Low vegetation": 2,
    "Impervious": 3,
    "Buildings": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing einer 1-km²-SwissSURFACE3D-Kachel."
    )
    parser.add_argument("--config", required=True, help="Pfad zur YAML-Konfiguration")
    parser.add_argument(
        "--tile-index",
        type=int,
        required=True,
        help="Index der 1-km²-COPC-Kachel aus der YAML-Datei",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Optional: nur einen 100×100-m-Chunk verarbeiten",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_tile_origin(filename):
    match = re.search(r"_(\d+)-(\d+)_2056_", filename)
    if match is None:
        raise ValueError(f"Kachelkoordinaten konnten nicht aus {filename} gelesen werden.")
    return int(match.group(1)) * 1000, int(match.group(2)) * 1000


def create_chunks(tile_x, tile_y):
    chunks = []
    chunk_id = 0
    for y0 in range(tile_y, tile_y + 1000, CHUNK_SIZE):
        for x0 in range(tile_x, tile_x + 1000, CHUNK_SIZE):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "xmin": x0,
                    "xmax": x0 + CHUNK_SIZE,
                    "ymin": y0,
                    "ymax": y0 + CHUNK_SIZE,
                }
            )
            chunk_id += 1
    return chunks


def build_lidar_inventory(lidar_dir):
    """Räumliche 1-km²-Ausdehnung aller vorhandenen COPC-Dateien."""
    inventory = []
    for path in sorted(lidar_dir.glob("*.copc.laz")):
        match = re.search(r"_(\d+)-(\d+)_2056_", path.name)
        if match is None:
            continue
        xmin = int(match.group(1)) * 1000
        ymin = int(match.group(2)) * 1000
        inventory.append(
            {
                "path": path,
                "xmin": xmin,
                "xmax": xmin + 1000,
                "ymin": ymin,
                "ymax": ymin + 1000,
            }
        )
    return inventory


def load_vector_data(cfg, tile_geom):
    """AV und Baumkronen einmal pro 1-km²-Job laden."""
    target = gpd.read_file(cfg["av"]["path"], layer=cfg["av"]["layer"])
    target = target[target["target_class"].notna()][["target_class", "geometry"]]
    target = target[target.intersects(tile_geom)].copy()

    trees = gpd.read_file(
        cfg["tree_canopy"]["path"],
        layer=cfg["tree_canopy"]["layer"],
    )[["geometry"]]
    trees = trees[trees.intersects(tile_geom)].copy()

    return target, trees


def read_chunk_points(lidar_path, chunk, crs):
    """100×100-m-Zielpunkte lesen; rechts/oben exklusiv (half-open)."""
    bounds = Bounds(
        mins=np.array([chunk["xmin"], chunk["ymin"]]),
        maxs=np.array([chunk["xmax"], chunk["ymax"]]),
    )

    with CopcReader.open(lidar_path) as reader:
        points = reader.query(bounds=bounds)

    x_all = np.asarray(points.x)
    y_all = np.asarray(points.y)
    keep = (
        (x_all >= chunk["xmin"])
        & (x_all < chunk["xmax"])
        & (y_all >= chunk["ymin"])
        & (y_all < chunk["ymax"])
    )

    x = x_all[keep]
    y = y_all[keep]

    return gpd.GeoDataFrame(
        {
            "point_id": np.arange(len(x)),
            "original_class": np.asarray(points.classification)[keep],
            "x": x,
            "y": y,
            "z": np.asarray(points.z)[keep],
            "intensity": np.asarray(points.intensity)[keep],
            "return_num": np.asarray(points.return_number)[keep],
            "num_returns": np.asarray(points.number_of_returns)[keep],
            "scan_angle": np.asarray(points.scan_angle)[keep],
        },
        geometry=gpd.points_from_xy(x, y),
        crs=crs,
    )


def add_spatial_context(gdf, chunk, target_tile, trees_tile):
    """AV-Klasse und Baumkronen-Zugehörigkeit wie im validierten Testworkflow."""
    gdf["av_class"] = None
    gdf["inside_tree_crown"] = False
    if gdf.empty:
        return gdf

    chunk_geom = box(chunk["xmin"], chunk["ymin"], chunk["xmax"], chunk["ymax"])
    mask_23 = gdf["original_class"].isin([2, 3])

    target = target_tile[target_tile.intersects(chunk_geom)]
    if not target.empty and mask_23.any():
        joined = gpd.sjoin(
            gdf.loc[mask_23, ["point_id", "geometry"]],
            target,
            how="left",
            predicate="within",
        ).drop_duplicates(subset="point_id")
        gdf["av_class"] = gdf["point_id"].map(
            dict(zip(joined["point_id"], joined["target_class"]))
        )

    trees = trees_tile[trees_tile.intersects(chunk_geom)]
    if not trees.empty:
        joined = gpd.sjoin(
            gdf[["point_id", "geometry"]],
            trees,
            how="left",
            predicate="within",
        )
        inside_ids = set(
            joined.loc[joined["index_right"].notna(), "point_id"]
        )
        gdf["inside_tree_crown"] = gdf["point_id"].isin(inside_ids)

    return gdf


def apply_rule(gdf, mask, label, status, basis):
    """Kleine Hilfsfunktion gegen wiederholte drei gdf.loc-Zeilen."""
    gdf.loc[mask, "final_label"] = label
    gdf.loc[mask, "status"] = status
    gdf.loc[mask, "rule_basis"] = basis


def relabel_points(gdf):
    """Relabeling-Regeln aus dem bereits validierten Prototyp."""
    gdf["final_label"] = None
    gdf["status"] = "unresolved"
    gdf["rule_basis"] = None

    removed = gdf["original_class"].isin([1, 14, 15, 27])
    gdf.loc[removed, "status"] = "removed"
    gdf.loc[removed, "rule_basis"] = "documented"

    # Direkte LiDAR-Zuordnung
    apply_rule(
        gdf,
        gdf["original_class"].isin([6, 26]) & ~removed,
        "Buildings",
        "direct_lidar",
        "documented",
    )
    apply_rule(
        gdf,
        (gdf["original_class"] == 9) & ~removed,
        "Water",
        "direct_lidar",
        "documented",
    )
    apply_rule(
        gdf,
        (gdf["original_class"] == 17) & ~removed,
        "Impervious",
        "direct_lidar",
        "documented",
    )

    # Dokumentierte AV-Kontextregeln
    apply_rule(
        gdf,
        (gdf["original_class"] == 2)
        & (gdf["av_class"] == "Tree canopy")
        & gdf["final_label"].isna(),
        "Low vegetation",
        "av_context",
        "documented",
    )
    apply_rule(
        gdf,
        (gdf["original_class"] == 3)
        & (gdf["av_class"] == "Tree canopy")
        & gdf["final_label"].isna(),
        "Tree canopy",
        "av_context",
        "documented",
    )
    apply_rule(
        gdf,
        (gdf["original_class"] == 2)
        & (gdf["av_class"] == "Impervious")
        & gdf["final_label"].isna(),
        "Impervious",
        "av_context",
        "documented",
    )

    # Eigene transparente Ergänzung
    apply_rule(
        gdf,
        (gdf["original_class"] == 2)
        & (gdf["av_class"] == "Low vegetation")
        & gdf["final_label"].isna(),
        "Low vegetation",
        "av_lowveg",
        "inferred",
    )

    # Baumkronen-Verfeinerung
    apply_rule(
        gdf,
        (gdf["original_class"] == 3)
        & gdf["inside_tree_crown"]
        & gdf["final_label"].isna(),
        "Tree canopy",
        "tree_refinement",
        "documented",
    )
    apply_rule(
        gdf,
        (gdf["original_class"] == 2)
        & gdf["inside_tree_crown"]
        & gdf["final_label"].isna(),
        "Low vegetation",
        "tree_refinement",
        "documented",
    )

    apply_rule(
        gdf,
        (gdf["original_class"] == 3)
        & (gdf["av_class"] == "Low vegetation")
        & (~gdf["inside_tree_crown"])
        & gdf["final_label"].isna(),
        "Low vegetation",
        "av_lowveg",
        "inferred",
    )

    unresolved = gdf["final_label"].isna() & ~removed
    gdf.loc[unresolved, "status"] = "unresolved_conflict"
    gdf.loc[unresolved, "rule_basis"] = "not_assigned"

    return gdf


def select_ground_files(chunk, buffer_m, inventory):
    xmin = chunk["xmin"] - buffer_m
    xmax = chunk["xmax"] + buffer_m
    ymin = chunk["ymin"] - buffer_m
    ymax = chunk["ymax"] + buffer_m

    files = [
        item["path"]
        for item in inventory
        if not (
            item["xmax"] <= xmin
            or item["xmin"] >= xmax
            or item["ymax"] <= ymin
            or item["ymin"] >= ymax
        )
    ]
    return xmin, xmax, ymin, ymax, files


def compute_height_agl(gdf, chunk, buffer_m, k_neighbors, inventory):
    """8-NN (konfigurierbar) + inverse Distanzgewichtung aus Ground-Klasse 2."""
    xmin, xmax, ymin, ymax, ground_files = select_ground_files(
        chunk, buffer_m, inventory
    )
    if not ground_files:
        raise RuntimeError("Keine COPC-Datei für den Ground-Puffer gefunden.")

    bounds = Bounds(
        mins=np.array([xmin, ymin]),
        maxs=np.array([xmax, ymax]),
    )
    x_parts, y_parts, z_parts = [], [], []

    for path in ground_files:
        with CopcReader.open(path) as reader:
            points = reader.query(bounds=bounds)

        gx = np.asarray(points.x)
        gy = np.asarray(points.y)
        gz = np.asarray(points.z)
        classes = np.asarray(points.classification)

        keep = (
            (classes == 2)
            & (gx >= xmin)
            & (gx < xmax)
            & (gy >= ymin)
            & (gy < ymax)
        )
        if np.any(keep):
            x_parts.append(gx[keep])
            y_parts.append(gy[keep])
            z_parts.append(gz[keep])

    if not x_parts:
        raise RuntimeError("Keine Ground-Punkte (Klasse 2) im Puffer gefunden.")

    ground_x = np.concatenate(x_parts)
    ground_y = np.concatenate(y_parts)
    ground_z = np.concatenate(z_parts)

    if len(ground_x) < k_neighbors:
        raise RuntimeError(
            f"Nur {len(ground_x)} Ground-Punkte gefunden, k={k_neighbors} benötigt."
        )

    tree = cKDTree(np.column_stack([ground_x, ground_y]))
    point_xy = np.column_stack([gdf["x"].to_numpy(), gdf["y"].to_numpy()])
    distances, indices = tree.query(point_xy, k=k_neighbors)

    neighbor_z = ground_z[indices]
    weights = 1.0 / np.maximum(distances, 0.01)
    ground_z_est = (
        np.sum(neighbor_z * weights, axis=1)
        / np.sum(weights, axis=1)
    )
    height_agl = gdf["z"].to_numpy() - ground_z_est

    gdf["ground_z_est"] = ground_z_est
    gdf["ground_distance_m"] = distances[:, 0]
    gdf["height_agl_m"] = height_agl

    info = {
        "ground_point_count": len(ground_x),
        "ground_file_count": len(ground_files),
        "ground_files": ";".join(path.name for path in ground_files),
        "ground_distance_median": float(np.median(distances[:, 0])),
        "ground_distance_p95": float(np.percentile(distances[:, 0], 95)),
        "height_agl_min": float(np.min(height_agl)),
        "height_agl_median": float(np.median(height_agl)),
        "height_agl_max": float(np.max(height_agl)),
    }
    return gdf, info


def save_labeled_chunk(gdf, output_dir, chunk_id):
    """Nur gelabelte Punkte speichern; ungelöste/entfernte Punkte bleiben draussen."""
    usable = gdf[gdf["final_label"].notna()]
    labels = usable["final_label"].map(LABEL_MAP).to_numpy(dtype=np.int8)

    path = output_dir / f"chunk_{chunk_id:03d}.npz"
    np.savez_compressed(
        path,
        x=usable["x"].to_numpy(dtype=np.float64),
        y=usable["y"].to_numpy(dtype=np.float64),
        height_agl_m=usable["height_agl_m"].to_numpy(dtype=np.float32),
        intensity=usable["intensity"].to_numpy(),
        return_num=usable["return_num"].to_numpy(),
        num_returns=usable["num_returns"].to_numpy(),
        scan_angle=usable["scan_angle"].to_numpy(),
        labels=labels,
    )
    return path, len(usable)


def summarize_chunk(gdf, chunk, tile_index, ground_info):
    final = gdf["final_label"].value_counts()
    status = gdf["status"].value_counts()
    basis = gdf["rule_basis"].value_counts()

    return {
        "tile_index": tile_index,
        "chunk_id": chunk["chunk_id"],
        "xmin": chunk["xmin"],
        "xmax": chunk["xmax"],
        "ymin": chunk["ymin"],
        "ymax": chunk["ymax"],
        "total_points": len(gdf),
        "labeled_points": int(gdf["final_label"].notna().sum()),
        "water": int(final.get("Water", 0)),
        "tree_canopy": int(final.get("Tree canopy", 0)),
        "low_vegetation": int(final.get("Low vegetation", 0)),
        "impervious": int(final.get("Impervious", 0)),
        "buildings": int(final.get("Buildings", 0)),
        "removed": int(status.get("removed", 0)),
        "unresolved_conflict": int(status.get("unresolved_conflict", 0)),
        "documented": int(basis.get("documented", 0)),
        "inferred": int(basis.get("inferred", 0)),
        "not_assigned": int(basis.get("not_assigned", 0)),
        **ground_info,
    }


def print_chunk_qa(gdf, chunk, ground_info):
    print("\nCHUNK-TEST")
    print("Chunk:", chunk["chunk_id"])
    print("Punkte:", len(gdf))

    print("\nOriginalklassen:")
    print(gdf["original_class"].value_counts().sort_index())

    print("\nAV-ZUORDNUNG:")
    print(gdf["av_class"].value_counts(dropna=False))
    print("\nPunkte innerhalb Baumkronen:", int(gdf["inside_tree_crown"].sum()))

    print("\nRELABEL-TEST")
    print("\nFinale Klassen:")
    print(gdf["final_label"].value_counts(dropna=False))
    print("\nStatus:")
    print(gdf["status"].value_counts())
    print("\nRegelgrundlage:")
    print(gdf["rule_basis"].value_counts(dropna=False))
    print("\nUngelöste Konflikte:")
    print(
        gdf.loc[
            gdf["status"] == "unresolved_conflict",
            ["original_class", "av_class"],
        ].value_counts()
    )

    print("\nGround-Punkte im 25-m-Puffer:", ground_info["ground_point_count"])
    print("\nHEIGHT-AGL-TEST")
    print("Nearest Ground Distance:")
    print("  Median:", round(ground_info["ground_distance_median"], 3), "m")
    print("  P95:", round(ground_info["ground_distance_p95"], 3), "m")
    print("\nHeight AGL:")
    print("  Min:", round(ground_info["height_agl_min"], 3), "m")
    print("  Median:", round(ground_info["height_agl_median"], 3), "m")
    print("  Max:", round(ground_info["height_agl_max"], 3), "m")


def process_chunk(
    lidar_path,
    chunk,
    cfg,
    target_tile,
    trees_tile,
    inventory,
    output_dir,
    tile_index,
    verbose=False,
):
    gdf = read_chunk_points(lidar_path, chunk, cfg["project"]["crs"])
    if gdf.empty:
        raise RuntimeError(f"Chunk {chunk['chunk_id']} enthält keine LiDAR-Punkte.")

    gdf = add_spatial_context(gdf, chunk, target_tile, trees_tile)
    gdf = relabel_points(gdf)
    gdf, ground_info = compute_height_agl(
        gdf,
        chunk,
        cfg["ground_height"]["buffer_m"],
        cfg["ground_height"]["k_neighbors"],
        inventory,
    )

    output_path, labeled_count = save_labeled_chunk(
        gdf, output_dir, chunk["chunk_id"]
    )
    summary = summarize_chunk(gdf, chunk, tile_index, ground_info)

    if verbose:
        print_chunk_qa(gdf, chunk, ground_info)
        print("\nGespeichert:", output_path)
        print("Gelabelte Punkte gespeichert:", labeled_count)

    return summary


def main():
    args = parse_args()
    cfg = load_config(args.config)

    lidar_files = cfg["lidar"]["files"]
    lidar_dir = Path(cfg["lidar"]["directory"])

    if not 0 <= args.tile_index < len(lidar_files):
        raise ValueError(
            f"tile-index muss zwischen 0 und {len(lidar_files) - 1} liegen."
        )

    filename = lidar_files[args.tile_index]
    lidar_path = lidar_dir / filename
    if not lidar_path.exists():
        raise FileNotFoundError(f"COPC-Datei nicht gefunden: {lidar_path}")

    tile_x, tile_y = parse_tile_origin(filename)
    tile_geom = box(tile_x, tile_y, tile_x + 1000, tile_y + 1000)
    chunks = create_chunks(tile_x, tile_y)

    if args.chunk_index is None:
        selected_chunks = chunks
    else:
        if not 0 <= args.chunk_index < len(chunks):
            raise ValueError(
                f"chunk-index muss zwischen 0 und {len(chunks) - 1} liegen."
            )
        selected_chunks = [chunks[args.chunk_index]]

    print("AV und Baumkronen einmalig laden ...")
    target_tile, trees_tile = load_vector_data(cfg, tile_geom)
    inventory = build_lidar_inventory(lidar_dir)
    if not inventory:
        raise RuntimeError("Keine COPC-Dateien im LiDAR-Verzeichnis gefunden.")

    tile_name = f"tile_{args.tile_index:02d}_{tile_x // 1000}-{tile_y // 1000}"
    output_dir = Path(cfg["output"]["intermediate"]) / tile_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project": cfg["project"]["name"],
        "crs": cfg["project"]["crs"],
        "source_copc": filename,
        "tile_index": args.tile_index,
        "tile_bounds": {
            "xmin": tile_x,
            "xmax": tile_x + 1000,
            "ymin": tile_y,
            "ymax": tile_y + 1000,
        },
        "chunk_size_m": CHUNK_SIZE,
        "half_open_boundaries": True,
        "ground_height": {
            "method": "k-NN inverse-distance weighted from SwissSURFACE3D Ground class 2",
            "k_neighbors": cfg["ground_height"]["k_neighbors"],
            "buffer_m": cfg["ground_height"]["buffer_m"],
        },
        "label_map": LABEL_MAP,
        "saved_fields": [
            "x",
            "y",
            "height_agl_m",
            "intensity",
            "return_num",
            "num_returns",
            "scan_angle",
            "labels",
        ],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nProjekt:", cfg["project"]["name"])
    print("Tile-Index:", args.tile_index)
    print("COPC:", lidar_path)
    print(
        "Kachel:",
        f"{tile_x}-{tile_x + 1000} / {tile_y}-{tile_y + 1000}",
    )
    print("Chunks zu verarbeiten:", len(selected_chunks))
    print("Output:", output_dir)

    summaries = []
    summary_path = output_dir / "chunk_summary.csv"

    for number, chunk in enumerate(selected_chunks, start=1):
        verbose = args.chunk_index is not None

        if not verbose:
            print(
                f"[{number:03d}/{len(selected_chunks):03d}] "
                f"Chunk {chunk['chunk_id']:03d} ..."
            )

        summary = process_chunk(
            lidar_path,
            chunk,
            cfg,
            target_tile,
            trees_tile,
            inventory,
            output_dir,
            args.tile_index,
            verbose=verbose,
        )
        summaries.append(summary)

        # Nach jedem Chunk aktualisieren: bei Abbruch bleibt der bisherige Stand erhalten.
        pd.DataFrame(summaries).to_csv(summary_path, index=False)

        if not verbose:
            print(
                f"    {summary['labeled_points']} gelabelt, "
                f"{summary['unresolved_conflict']} ungelöst"
            )

    print("\nPREPROCESSING ERFOLGREICH")
    print("Zusammenfassung:", summary_path)


if __name__ == "__main__":
    main()
