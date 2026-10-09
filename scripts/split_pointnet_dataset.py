"""Raeumlicher Split: festgelegte Variante B fuer Waedenswil."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
RATIOS = np.array([0.70, 0.15, 0.15])


def choose_split(coords, histograms):
    """Feste, vom Nutzer ausgewaehlte Grenzen; untere inklusive, obere exklusive."""
    rectangles = {
        "val": [2691100, 2693000, 1232100, 1232400],
        "test": [2691200, 2692600, 1232400, 1232800],
    }
    assignment = np.zeros(len(coords), dtype=np.int8)
    for index, split in enumerate(SPLITS[1:], 1):
        xmin, xmax, ymin, ymax = rectangles[split]
        mask = ((coords[:, 0] >= xmin) & (coords[:, 0] < xmax)
                & (coords[:, 1] >= ymin) & (coords[:, 1] < ymax))
        if (assignment[mask] != 0).any():
            raise RuntimeError("Split-Gebiete ueberlappen sich.")
        assignment[mask] = index
    counts = np.bincount(assignment, minlength=3)
    if counts.tolist() != [4302, 912, 896]:
        raise RuntimeError(f"Variante B erwartet 4302/912/896 Tiles; gefunden: {counts.tolist()}")
    for index in range(3):
        if (histograms[assignment == index].sum(axis=0) == 0).any():
            raise RuntimeError("Mindestens eine Klasse fehlt in einem Split.")
    return assignment, dict(variant="B", rectangles_xyxy=rectangles,
                           rectangle_order="xmin, xmax, ymin, ymax",
                           inclusion="lower inclusive, upper exclusive",
                           training="all remaining valid tiles",
                           fractions=(counts / len(coords)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.input_dir), Path(args.output_dir)
    with (source / "metadata.json").open() as f:
        meta = json.load(f)
    with (source / "train_tiles.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    features = np.load(source / "train_features.npy", mmap_mode="r")
    labels = np.load(source / "train_labels.npy", mmap_mode="r")
    n, target, nf = len(rows), int(meta["points_per_tile"]), int(meta["num_features"])
    nc = int(meta["num_classes"])
    names = [meta["classes"][str(i)] for i in range(nc)]
    if features.shape != (n, target, nf) or labels.shape != (n, target):
        raise RuntimeError("Array-Form passt nicht zur Tile-Tabelle.")
    if features.dtype != np.float32 or labels.dtype != np.int64:
        raise RuntimeError("Unerwartete Datentypen.")
    if [int(r["array_index"]) for r in rows] != list(range(n)):
        raise RuntimeError("Tile-Tabelle nicht in Array-Reihenfolge.")
    coords = np.array([[int(r["xmin"]), int(r["ymin"])] for r in rows])
    size = int(meta["tile_size_m"])
    if size <= 0 or 100 % size or (coords % size).any():
        raise RuntimeError("Tile-Raster nicht mit 100-m-Grenzen kompatibel.")
    if len(np.unique(coords, axis=0)) != n:
        raise RuntimeError("Doppelte Tile-Koordinaten.")
    histograms = np.array([[int(r[f"sampled_{name}"]) for name in names] for r in rows])
    if (histograms < 0).any() or not (histograms.sum(axis=1) == target).all():
        raise RuntimeError("Ungueltige Klassenhaeufigkeiten im Inventar.")
    assignment, spatial = choose_split(coords, histograms)
    print("Raeumliche Aufteilung:", json.dumps(spatial), flush=True)
    staging = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"Ausgabe existiert: {output}")
    staging.mkdir(parents=True, exist_ok=False)
    split_counts, class_counts, mapped = {}, {}, []
    for split_index, split in enumerate(SPLITS):
        source_indices = np.flatnonzero(assignment == split_index)
        count = len(source_indices)
        split_counts[split] = count
        out_f = np.lib.format.open_memmap(staging / f"{split}_features.npy", mode="w+",
                                          dtype=np.float32, shape=(count, target, nf))
        out_l = np.lib.format.open_memmap(staging / f"{split}_labels.npy", mode="w+",
                                          dtype=np.int64, shape=(count, target))
        total_classes = np.zeros(nc, dtype=np.int64)
        split_rows = []
        for dest, original in enumerate(source_indices):
            f, lab = features[original], labels[original]
            if not np.isfinite(f).all() or not np.isin(lab, range(nc)).all():
                raise RuntimeError(f"Ungueltige Werte in Quell-Tile {original}.")
            histogram = np.bincount(lab, minlength=nc)
            if not np.array_equal(histogram, histograms[original]):
                raise RuntimeError(f"Klassen stimmen nicht mit Tabelle ueberein: {original}.")
            out_f[dest], out_l[dest] = f, lab
            total_classes += histogram
            row = dict(rows[original])
            row.update(array_index=dest, source_array_index=int(original), split=split)
            x, y = coords[original]
            row["wkt"] = f"POLYGON (({x} {y}, {x+size} {y}, {x+size} {y+size}, {x} {y+size}, {x} {y}))"
            split_rows.append(row)
            if (dest + 1) % 500 == 0:
                print(f"{split}: {dest+1}/{count} Tiles geschrieben", flush=True)
        out_f.flush()
        out_l.flush()
        del out_f, out_l
        mapped.extend(split_rows)
        class_counts[split] = total_classes.tolist()
        with (staging / f"{split}_tiles.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(split_rows[0]))
            w.writeheader()
            w.writerows(split_rows)
        print(f"{split}: {count} Tiles ({100*count/n:.2f} %)", flush=True)
        for name, value in zip(names, total_classes):
            print(f"  {name}: {value:,} ({100*value/(count*target):.2f} %)", flush=True)
    with (staging / "split_tiles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mapped[0]))
        w.writeheader()
        w.writerows(mapped)
    result_meta = dict(points_per_tile=target, num_features=nf, num_classes=nc,
                       features=meta["features"], classes=meta["classes"], tile_size_m=size,
                       complete=True, available_splits=list(SPLITS), split_sizes=split_counts,
                       class_counts=class_counts, target_ratios=RATIOS.tolist(),
                       split_method="fixed_variant_B_rectangles_on_100m_grid",
                       spatial_split=spatial, buffer_m=0, crs="EPSG:2056",
                       source_dir=str(source.resolve()), source_metadata=meta,
                       height_policy=meta["height_policy"])
    with (staging / "metadata.json").open("w") as f:
        json.dump(result_meta, f, indent=2)
    if output.exists():
        raise FileExistsError(f"Ausgabe inzwischen vorhanden: {output}")
    staging.rename(output)
    print("\nRAEUMLICHER SPLIT ERFOLGREICH")
    print("Ausgabe:", output)
    print("QGIS: split_tiles.csv als WKT-Polygon laden, CRS EPSG:2056, nach split einfaerben.")
    print("Teilflaechen koennen sich an Grenzen beruehren; kein Abstandspuffer.")


if __name__ == "__main__":
    main()
