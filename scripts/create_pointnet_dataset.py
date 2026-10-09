"""25-m-Inventar und optionaler Sampling-Zwischenexport (noch ohne Datensplit)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


CLASS_NAMES = ["water", "tree_canopy", "low_vegetation", "impervious", "buildings"]
FEATURE_NAMES = ["x_local_m", "y_local_m", "height_agl_m", "intensity",
                 "return_num", "num_returns", "scan_angle"]


def sample_indices(n, target, rng):
    """Bei kleinen Tiles jeden Originalpunkt behalten, nur Rest auffuellen."""
    if n <= 0 or target <= 0:
        raise ValueError("Punktzahlen muessen positiv sein.")
    if n > target:
        indices = rng.choice(n, size=target, replace=False)
    else:
        indices = np.arange(n, dtype=np.int64)
        if n < target:
            indices = np.concatenate([
                indices, rng.choice(n, size=target - n, replace=True)
            ])
    rng.shuffle(indices)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="Inventar-CSV")
    parser.add_argument("--sample-dir", help="Optional: neuer Ordner fuer Zwischenexport")
    parser.add_argument("--min-points", type=int, default=4000)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["output"]["intermediate"])
    size = int(cfg["tiling"]["tile_size_m"])
    target = int(cfg["tiling"]["points_per_tile"])
    seed = int(cfg["tiling"].get("seed", 42))
    if size <= 0 or not 1 <= args.min_points <= target or seed < 0:
        raise ValueError("Ungueltige Tile-Groesse, Mindestpunktzahl, Zielpunktzahl oder Seed.")
    summaries = sorted(root.glob("tile_*/chunk_summary.csv"))
    if len(summaries) != len(cfg["lidar"]["files"]) or not summaries:
        raise RuntimeError("Anzahl der Kachel-Zusammenfassungen stimmt nicht.")
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    if sample_dir is not None:
        # Ein abgebrochener oder alter Export darf nicht unbemerkt vermischt werden.
        sample_dir.mkdir(parents=True, exist_ok=False)

    records = []
    seen_origins = set()
    exported = 0
    sampled_missing = 0
    for source_number, summary in enumerate(summaries):
        with summary.open(newline="") as f:
            chunks = list(csv.DictReader(f))
        seen_chunks = set()
        for chunk in chunks:
            chunk_id = int(chunk["chunk_id"])
            if chunk_id in seen_chunks:
                raise RuntimeError(f"Doppelte Chunk-ID: {summary}, {chunk_id}")
            seen_chunks.add(chunk_id)
            xmin, xmax = int(chunk["xmin"]), int(chunk["xmax"])
            ymin, ymax = int(chunk["ymin"]), int(chunk["ymax"])
            nx, ny = (xmax - xmin) // size, (ymax - ymin) // size
            if nx <= 0 or ny <= 0 or nx * size != xmax - xmin or ny * size != ymax - ymin:
                raise RuntimeError("Tile-Groesse passt nicht zur Chunk-Groesse.")
            counts = np.zeros(nx * ny, dtype=np.int64)
            missing = counts.copy()
            classes = np.zeros((nx * ny, 5), dtype=np.int64)
            path = summary.parent / f"chunk_{chunk_id:03d}.npz"
            arrays = None
            if int(chunk["total_points"]) > 0:
                keys = ["x", "y", "labels", "height_agl_m"]
                if sample_dir is not None:
                    keys += FEATURE_NAMES[3:]
                with np.load(path, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in keys}
                x, y, labels = arrays["x"], arrays["y"], arrays["labels"]
                if any(a.ndim != 1 or len(a) != len(x) for a in arrays.values()):
                    raise RuntimeError(f"Unpassende Array-Formen: {path}")
                if not (((x >= xmin) & (x < xmax)).all() and
                        ((y >= ymin) & (y < ymax)).all()):
                    raise RuntimeError(f"Punkte ausserhalb des Chunks: {path}")
                if not np.isin(labels, range(5)).all():
                    raise RuntimeError(f"Unbekannte Labels: {path}")
                labels = labels.astype(np.int64)
                for key, values in arrays.items():
                    invalid = np.isinf(values) if key == "height_agl_m" else ~np.isfinite(values)
                    if invalid.any():
                        raise RuntimeError(f"Ungueltige Werte in {key}: {path}")
                ix = np.floor((x - xmin) / size).astype(np.int64)
                iy = np.floor((y - ymin) / size).astype(np.int64)
                ids = iy * nx + ix
                counts = np.bincount(ids, minlength=nx * ny)
                missing = np.bincount(ids[np.isnan(arrays["height_agl_m"])], minlength=nx * ny)
                classes = np.bincount(ids * 5 + labels, minlength=nx * ny * 5).reshape(nx * ny, 5)
            if int(counts.sum()) != int(chunk["labeled_points"]):
                raise RuntimeError(f"Punktzahl stimmt nicht mit Chunk-CSV ueberein: {path}")

            for i in range(nx * ny):
                tx, ty = xmin + (i % nx) * size, ymin + (i // nx) * size
                if (tx, ty) in seen_origins:
                    raise RuntimeError(f"Doppeltes Tile: {tx}, {ty}")
                seen_origins.add((tx, ty))
                n = int(counts[i])
                status = "empty" if n == 0 else "below_minimum" if n < args.min_points else "eligible"
                row = dict(source_tile=summary.parent.name, chunk_id=chunk_id,
                           xmin=tx, ymin=ty, points=n, missing_height=int(missing[i]),
                           enough_for_target=int(n >= target), status=status,
                           sample_file="", sampled_points=0, repeated_points=0,
                           sampled_missing_height=0)
                row.update({name: int(classes[i, j]) for j, name in enumerate(CLASS_NAMES)})
                if sample_dir is not None and status == "eligible":
                    # Eigener Zufallsstrom je Tile; keine Abhaengigkeit von vorherigen Auswahlen.
                    rng = np.random.default_rng(np.random.SeedSequence([seed, source_number, chunk_id, i]))
                    source_indices = np.flatnonzero(ids == i)[sample_indices(n, target, rng)]
                    features = np.column_stack([
                        arrays["x"][source_indices] - tx,
                        arrays["y"][source_indices] - ty,
                        *[arrays[key][source_indices] for key in FEATURE_NAMES[2:]]
                    ]).astype(np.float32)
                    chosen_labels = labels[source_indices]
                    if np.isinf(features).any() or not np.isfinite(features[:, [0, 1, 3, 4, 5, 6]]).all():
                        raise RuntimeError(f"Ungueltige Features nach Konvertierung: {tx}, {ty}")
                    filename = f"tile_{tx}_{ty}.npz"
                    nan_count = int(np.isnan(features[:, 2]).sum())
                    np.savez_compressed(sample_dir / filename, features=features,
                                        labels=chosen_labels, source_indices=source_indices,
                                        origin_xy=np.array([tx, ty], dtype=np.float64))
                    row.update(status="sampled", sample_file=filename,
                               sampled_points=target, repeated_points=max(0, target - n),
                               sampled_missing_height=nan_count)
                    for j in range(5):
                        row[f"sampled_{CLASS_NAMES[j]}"] = int((chosen_labels == j).sum())
                    exported += 1
                    sampled_missing += nan_count
                for name in CLASS_NAMES:
                    row.setdefault(f"sampled_{name}", 0)
                records.append(row)
        print(f"{summary.parent.name}: verarbeitet", flush=True)

    if not records:
        raise RuntimeError("Keine Tiles gefunden.")
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
    for threshold in sorted(set([1, 1000, 4000, 8000, target, args.min_points])):
        excluded = [row for row in records if row["points"] < threshold]
        print(f"\nMinimum {threshold}: {len(records) - len(excluded)} Tiles verwendbar")
        for name in CLASS_NAMES:
            total = sum(row[name] for row in records)
            lost = sum(row[name] for row in excluded)
            print(f"  {name}: {lost:,} Punkte ausgeschlossen ({100 * lost / total if total else 0:.1f} %)")
    if sample_dir is not None:
        metadata = dict(stage="sampled_intermediate", complete=True,
                        ready_for_training=False, split=None,
                        missing_height_policy="preserve_nan",
                        source_root=str(root.resolve()), inventory_csv=str(output.resolve()),
                        features=FEATURE_NAMES, class_names=CLASS_NAMES,
                        tile_size_m=size, points_per_tile=target, min_points=args.min_points,
                        seed=seed, numpy_version=np.__version__, tiles=exported,
                        sampled_points=exported * target,
                        sampled_missing_height=sampled_missing,
                        source_indices_meaning="Zero-based row indices in original chunk NPZ")
        with (sample_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSAMPLING-ZWISCHENEXPORT ERFOLGREICH: {exported} Tiles")
        print(f"Fehlende Hoehen nach Sampling (einschliesslich Wiederholungen): {sampled_missing:,}")
        print("Noch keine fertigen Trainingsdaten: Hoehenbehandlung und Datensplit offen.")
    print("\nTILE-INVENTAR ERFOLGREICH")
    print("CSV:", output)


if __name__ == "__main__":
    main()
