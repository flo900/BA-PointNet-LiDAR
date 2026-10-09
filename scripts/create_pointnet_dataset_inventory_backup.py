"""Erste Etappe: 25-m-Tiles inventarisieren, noch keine Trainingsdaten erstellen."""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["output"]["intermediate"])
    size = int(cfg["tiling"]["tile_size_m"])
    target = int(cfg["tiling"]["points_per_tile"])
    class_names = [
        "water", "tree_canopy", "low_vegetation", "impervious", "buildings"
    ]

    summaries = sorted(root.glob("tile_*/chunk_summary.csv"))
    if len(summaries) != len(cfg["lidar"]["files"]):
        raise RuntimeError("Anzahl der Kachel-Zusammenfassungen stimmt nicht.")

    records = []
    for summary in summaries:
        with summary.open(newline="") as f:
            chunks = list(csv.DictReader(f))

        for chunk in chunks:
            xmin, xmax = int(chunk["xmin"]), int(chunk["xmax"])
            ymin, ymax = int(chunk["ymin"]), int(chunk["ymax"])
            nx = (xmax - xmin) // size
            ny = (ymax - ymin) // size
            if nx * size != xmax - xmin or ny * size != ymax - ymin:
                raise RuntimeError("Tile-Grösse passt nicht zur Chunk-Grösse.")

            counts = np.zeros(nx * ny, dtype=np.int64)
            missing = counts.copy()
            classes = np.zeros((nx * ny, 5), dtype=np.int64)

            if int(chunk["total_points"]) > 0:
                path = summary.parent / f"chunk_{int(chunk['chunk_id']):03d}.npz"
                with np.load(path, allow_pickle=False) as data:
                    x, y = data["x"], data["y"]
                    labels = data["labels"].astype(np.int64)
                    if not (
                        ((x >= xmin) & (x < xmax)).all()
                        and ((y >= ymin) & (y < ymax)).all()
                    ):
                        raise RuntimeError(f"Punkte ausserhalb des Chunks: {path}")
                    if not np.isin(labels, range(5)).all():
                        raise RuntimeError(f"Unbekannte Labels: {path}")

                    ix = np.floor((x - xmin) / size).astype(np.int64)
                    iy = np.floor((y - ymin) / size).astype(np.int64)
                    ids = iy * nx + ix
                    counts = np.bincount(ids, minlength=nx * ny)
                    missing = np.bincount(
                        ids[np.isnan(data["height_agl_m"])],
                        minlength=nx * ny,
                    )
                    classes = np.bincount(
                        ids * 5 + labels, minlength=nx * ny * 5
                    ).reshape(nx * ny, 5)

            if int(counts.sum()) != int(chunk["labeled_points"]):
                raise RuntimeError("Punktzahl stimmt nicht mit Chunk-CSV überein.")

            for iy in range(ny):
                for ix in range(nx):
                    i = iy * nx + ix
                    row = {
                        "source_tile": summary.parent.name,
                        "chunk_id": int(chunk["chunk_id"]),
                        "xmin": xmin + ix * size,
                        "ymin": ymin + iy * size,
                        "points": int(counts[i]),
                        "missing_height": int(missing[i]),
                        "enough_for_target": int(counts[i] >= target),
                    }
                    row.update({
                        name: int(classes[i, j])
                        for j, name in enumerate(class_names)
                    })
                    records.append(row)

        print(f"{summary.parent.name}: inventarisiert", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    points = np.array([row["points"] for row in records])
    print(f"\nTiles insgesamt: {len(records)}")
    print(f"Leere Tiles: {int((points == 0).sum())}")
    print(f"Nicht leere Tiles unter {target}: {int(((points > 0) & (points < target)).sum())}")
    print(f"Tiles mit mindestens {target}: {int((points >= target).sum())}")
    print(f"Gelabelte Punkte insgesamt: {int(points.sum()):,}")

    print("\nVergleich möglicher Mindestpunktzahlen (noch keine Auswahl):")
    for threshold in sorted(set([1, 1000, 4000, 8000, target])):
        excluded = [row for row in records if row["points"] < threshold]
        kept = len(records) - len(excluded)
        print(f"\nMinimum {threshold}: {kept} Tiles verwendbar")
        for name in class_names:
            total = sum(row[name] for row in records)
            lost = sum(row[name] for row in excluded)
            percent = 100 * lost / total if total else 0
            print(f"  {name}: {lost:,} Punkte ausgeschlossen ({percent:.1f} %)")

    print("\nTILE-INVENTAR ERFOLGREICH")
    print("CSV:", output)


if __name__ == "__main__":
    main()
